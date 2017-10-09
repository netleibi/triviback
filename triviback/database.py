import os

import logging

logger = logging.getLogger(__name__)

class AbstractDatabase(object):

    name = None
    
    def __init__(self, conf=None):
        raise NotImplementedError()
    
    def close(self):
        pass
    
    def compact(self):
        pass
    
    def length(self):
        raise NotImplementedError()
    
    def put(self, k, v):
        raise NotImplementedError()

    def get(self, k):
        raise NotImplementedError()

    def delete(self, k):
        raise NotImplementedError()
    
    def list(self):
        raise NotImplementedError()
    
    def savepoint(self):
        pass


class MemoryDB(AbstractDatabase):
    
    name = 'memory'
    
    def __init__(self, conf=None):
        self._dict = dict()
        self._conf = conf
        
    def put(self, k, v):
        self._dict[k] = v
        
    def get(self, k):
        return self._dict.get(k, None)
    
    def delete(self, k):
        del self._dict[k]
        
    def length(self):
        return len(self._dict)
    
    def list(self):
        return list(self._dict)
    
    def dump(self):
        return self._dict
    
    def copy(self):
        memory_db = MemoryDB(self._conf)
        memory_db._dict = self._dict.copy()
        return memory_db

try:
    import semidbm
except ImportError:
    logger.warning('SemiDBM database not available due to missing library: semidbm')
else:
    class SemiDBM(AbstractDatabase):
        
        name = 'semidbm'
        
        def __init__(self, conf=None):
            if conf is None:
                raise Exception('Path required.')
            if not os.path.exists(os.path.dirname(conf)):
                os.makedirs(os.path.dirname(conf))
            self._db = semidbm.open(conf, 'c')
            
        def put(self, k, v):
            self._db[k] = v
            
        def get(self, k):
            db = self._db
            return db[k] if k in db else None
        
        def delete(self, k):
            del self._db[k]
    
        def close(self):
            return self._db.close()
    
        def compact(self):
            return self._db.compact()
        
        def length(self):
            return len(self._db)
        
        def list(self):
            return self._db.keys()

try:
    from ZODB import FileStorage, DB
    import BTrees.OOBTree
    import transaction
except ImportError:
    logger.warning('ZODB database not available due to missing library: ZODB')
else:
    class ZODB(AbstractDatabase):
        
        name = 'zodb'
        
        def __init__(self, conf=None):
            if conf is None:
                raise Exception('Path required.')
            if not os.path.exists(conf):
                os.makedirs(conf)
            storage = FileStorage.FileStorage(os.path.join(conf, 'db'), pack_keep_old=False)
            self._tmp_path = os.path.join(conf, 'db.tmp')
            self._db = DB(storage)
            self._connection = self._db.open()
            self._root = self._connection.root()
            if getattr(self._root, 'db', None) is None:
                self._root.db = BTrees.OOBTree.BTree()
            self._root_db = self._root.db
            self._transaction = transaction
            self._bytes_written = 0
            
        def put(self, k, v):
            self._root_db[k] = v
            self._bytes_written += len(k) + len(v)
            if self._bytes_written >= 104857600:
                self.compact()
            
        def get(self, k):
            db = self._root_db
            return db[k] if k in db else None
        
        def delete(self, k):
            del self._root_db[k]
    
        def close(self):
            self._transaction.commit()
            self._db.close()
            try:
                os.remove(self._tmp_path)
            except:
                pass
    
        def compact(self):
            self._transaction.commit()
            self._db.pack()
            self._bytes_written = 0
        
        def length(self):
            return len(self._root_db)
        
        def list(self):
            return self._root_db.keys()
        
        def savepoint(self):
            self._transaction.commit()

try:
    import sqlite3
except ImportError:
    logger.warning('SQLiteDB database not available due to missing library: sqlite3')
else:
    class SQLiteDB(AbstractDatabase):
        
        name = 'sqlite'
        
        def __init__(self, conf=None):
            if conf is None:
                raise Exception('Path required.')
            if not os.path.exists(os.path.dirname(conf)):
                os.makedirs(os.path.dirname(conf))
    
            self._binary_type = sqlite3.Binary
            self._db = sqlite3.connect(conf)
            self._db.execute('PRAGMA auto_vacuum = 0')
            
            self._cur = self._db.cursor()
            self._cur.execute('CREATE TABLE kvs(k TEXT, v TEXT)')
            
        def put(self, k, v):
            self._cur.execute('INSERT INTO kvs (k,v) SELECT ?, ? WHERE NOT EXISTS(SELECT 1 FROM kvs WHERE k=?)', (self._binary_type(k), self._binary_type(v), self._binary_type(k)))
            self._cur.execute('UPDATE kvs SET v=? WHERE k=?', (self._binary_type(v), self._binary_type(k)))
            
        def get(self, k):
            self._cur.execute('SELECT v FROM kvs WHERE k=?', (self._binary_type(k), ))
            result = self._cur.fetchone()
            return result if result is None else bytes(result[0])
        
        def delete(self, k):
            self._cur.execute('DELETE FROM kvs WHERE k=?', (self._binary_type(k), ))
    
        def close(self):
            return self._db.close()
    
        def compact(self):
            pass
        
        def length(self):
            pass
        
        def list(self):
            self._cur.execute('SELECT k FROM kvs')
            return map(bytes, self._cur.fetchall())
