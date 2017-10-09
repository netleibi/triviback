import base64
import hashlib
from multiprocessing import Process
import os
import random
import shutil
import struct
import tempfile
import unittest

from flask.app import Flask
import msgpack

from triviback import server, Triviback

#@unittest.skip('not now')
class AbstractTrivibackTest():
    '''
    Abstract super class for all test cases. Implements some helper methods.
    '''

    class WorkloadGenerator():

        ACTIONS = ['mkdir', 'createfile', 'modifyfile', 'deletefile', 'deletedir', 'noop']
        
        def __init__(self, source_path):
            self.source_path = source_path

            # initialize current state consisting only of the root directory
            self.directories = ['']
            self.files = []
            self.action_no = 0
            
        def random_action(self):
            self.action_no += 1

            source_path = self.source_path
            directories = self.directories
            files = self.files
            action_no = self.action_no
            
            # perform random action
            action = AbstractTrivibackTest.WorkloadGenerator.ACTIONS[random.randint(0, len(AbstractTrivibackTest.WorkloadGenerator.ACTIONS)-1)]
            if action == 'mkdir':
                # create directory in randomly chosen directory
                basedir = directories[random.randint(0, len(directories)-1)]
                name = os.path.join(basedir, 'dir{i}'.format(i=action_no))
                os.mkdir(os.path.join(source_path, name))
                directories.append(name)
            elif action == 'createfile':
                # create file in randomly chosen directory
                basedir = directories[random.randint(0, len(directories)-1)]
                name = os.path.join(basedir, 'file{i}'.format(i=action_no))
                with open(os.path.join(source_path, name), 'wb') as f:
                    f.write(os.urandom(random.randint(0, 10240)))
                files.append(name)
            elif action == 'modifyfile':
                # modify randomly chosen file (if any exists) at randomly chosen position
                if len(files) == 0:
                    return
                name = files[random.randint(0, len(files)-1)]
                with open(os.path.join(source_path, name), 'r+b') as f:
                    f.seek(random.randint(0, os.fstat(f.fileno()).st_size))
                    f.write(os.urandom(random.randint(0, 10240)))
            elif action == 'deletefile':
                # delete randomly chosen file (if any exists)
                if len(files) == 0:
                    return
                name = files[random.randint(0, len(files)-1)]
                os.remove(os.path.join(source_path, name))
                files.remove(name)
            elif action == 'deletedir':
                # delete randomly chosen directory (except root) recursively
                if len(directories) <= 1:
                    return
                name = os.path.join(directories[random.randint(1, len(directories)-1)], '')
                shutil.rmtree(os.path.join(source_path, name))
                files[:] = filter(lambda n: not os.path.join(n, '').startswith(os.path.join(name)), files)
                directories[:] = filter(lambda n: not os.path.join(n, '').startswith(os.path.join(name)), directories)
            elif action == 'noop':
                pass
            else:
                raise NotImplementedError()

    def setUp(self):
        self.cleanup_files = []
        self.cleanup_directories = []
    
    def tearDown(self):
        for filename in self.cleanup_files:
            try:
                os.remove(filename)
            except:
                pass
        for directory in self.cleanup_directories:
            try:
                shutil.rmtree(directory)
            except:
                pass

    def hashfile(self, path):
        # helper function that computes a file's sha1 hash
        BLOCKSIZE = 65536
        hasher = hashlib.sha1()
        with open(path, 'rb') as f:
            buf = f.read(BLOCKSIZE)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(BLOCKSIZE)
        return hasher.hexdigest()

    def compare_directories(self, path1, path2):
        # determine whether the contents of two directories are identical
        for entry1, entry2 in zip(sorted(os.listdir(path1)), sorted(os.listdir(path2))):
            # compare entry-by-entry with respect to name, type (dir/file), and stat information
            subpath1, subpath2 = os.path.join(path1, entry1), os.path.join(path2, entry2)
            lstat1, lstat2 = os.lstat(subpath1), os.lstat(subpath2)
            self.assertEqual(
                (entry1, os.path.isdir(subpath1), os.path.isfile(subpath1), lstat1.st_size, lstat1.st_mode, lstat1.st_uid, lstat1.st_gid, int(lstat1.st_mtime)),
                (entry2, os.path.isdir(subpath2), os.path.isfile(subpath2), lstat2.st_size, lstat2.st_mode, lstat2.st_uid, lstat2.st_gid, int(lstat2.st_mtime))
            )
            
            if os.path.isdir(subpath1):
                # compare subdirectories recursively
                self.compare_directories(subpath1, subpath2)
            elif os.path.isfile(subpath1):
                # compare file contents
                self.assertEqual(self.hashfile(subpath1), self.hashfile(subpath2), 'Content of file {subpath1} differs from content of file {subpath2}'.format(subpath1=subpath1, subpath2=subpath2))

    def _get_temp_directory(self):
        # returns a temporary directory that is automatically cleaned up at the end
        path = tempfile.mkdtemp()
        self.cleanup_directories.append(path)
        return path
    
    def _get_temp_file(self):
        # returns a temporary file that is automatically cleaned up at the end
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.cleanup_files.append(path)
        return path

'''
Implementation of test cases for the __init__ client that do not involve synchronization with backends.
'''

#@unittest.skip('not now')
class AbstractTrivibackClientOnlyTest(AbstractTrivibackTest):
    
    #@unittest.skip('not now')
    def testBackupAndRecoverEmptyDirectory(self):
        # create directories for backup and recovery
        backup_path = self._get_temp_directory()
        recover_path = self._get_temp_directory()
        
        # create backup
        backup_id = self.triviback.backup_path(backup_path)
        
        # recover backup and check consistency
        self.triviback.recover_path_by_id(backup_id, recover_path)
        self.compare_directories(backup_path, recover_path)

        # delete backup
        self.triviback.delete_backup_by_id(backup_id)

    #@unittest.skip('not now')
    def testBackupAndRecoverDirectoryWithRandomFiles(self):
        # create directories for backup and recovery
        backup_path = self._get_temp_directory()
        recover_path = self._get_temp_directory()
        
        # create some random files
        for i in range(0, 20):
            with open(os.path.join(backup_path, 'file{i}'.format(i=i)), 'wb') as f:
                f.write(os.urandom(random.randint(0, 10240)))
        
        # create backup
        backup_id = self.triviback.backup_path(backup_path)

        # recover backup and check consistency
        self.triviback.recover_path_by_id(backup_id, recover_path)
        self.compare_directories(backup_path, recover_path)

        # delete backup
        self.triviback.delete_backup_by_id(backup_id)

    #@unittest.skip('not now')
    def testBackupAndRecoverDirectoryWithRandomFilesAndDirectories(self):
        # create directories for backup and recovery
        backup_path = self._get_temp_directory()
        recover_path = self._get_temp_directory()
        
        # create some random files and directories
        directories = ['']
        for i in range(0, 100):
            # create file or directory in randomly chosen directory
            basedir = directories[random.randint(0, len(directories)-1)]
            if random.randint(0, 2) == 1:
                # with probability 1/3 create a directory
                name = os.path.join(basedir, 'dir{i}'.format(i=i))
                os.mkdir(os.path.join(backup_path, name))
                directories.append(name)
            else:
                # with probability 2/3 create random file
                with open(os.path.join(backup_path, basedir, 'file{i}'.format(i=i)), 'wb') as f:
                    f.write(os.urandom(random.randint(0, 10240)))
        
        # create backup
        backup_id = self.triviback.backup_path(backup_path)
        
        # recover backup and check consistency
        self.triviback.recover_path_by_id(backup_id, recover_path)
        self.compare_directories(backup_path, recover_path)
        
        # delete backup
        self.triviback.delete_backup_by_id(backup_id)

    #@unittest.skip('not now')
    def testComplexBackupAndRecoveryScenario(self):
        # create directories for backup and recovery
        source_path = self._get_temp_directory()

        # remember backups
        backups = []
        
        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                backup_id = self.triviback.backup_path(source_path)
                backups.append((backup_id, backup_path))
                
        # recover and delete backups in random order
        random.shuffle(backups)
        for (backup_id, backup_path) in backups:
            recover_path = self._get_temp_directory()
            self.triviback.recover_path_by_id(backup_id, recover_path)
            self.compare_directories(backup_path, recover_path)
            self.triviback.delete_backup_by_id(backup_id)

'''
The test cases for the __init__ client should be executed for each available database implementation.
'''

#@unittest.skip('not now')
class TrivibackMemoryDatabaseTest(AbstractTrivibackClientOnlyTest, unittest.TestCase):

    def setUp(self):
        super(TrivibackMemoryDatabaseTest, self).setUp()
        self.triviback = Triviback()
        
    def tearDown(self):
        # verify that backend is almost empty after execution of backup scenarios
        self.assertLessEqual(self.triviback.db.length(), 5) # root key + empty backup list + rc
        del self.triviback
        super(TrivibackMemoryDatabaseTest, self).tearDown()

    def testScrub(self):
        # performed on a consistent backup database, scrub must not change anything
        source_path = self._get_temp_directory()

        # remember backups
        backups = []

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                backup_id = self.triviback.backup_path(source_path)
                backups.append((backup_id, backup_path))
        
        # work with triviback db
        db = self.triviback.db
        original_db = self.triviback.db.copy()
        
        # delete all reference counters
        rc_keys = set()
        for key in self.triviback.db.list():
            if len(key) % 2 == 1 and bytes((key[0], )) == self.triviback.seccs_f._reference_counter.prefix:
                rc_keys.add(key)
        for rc_key in rc_keys:
            db.delete(rc_key)
        
        # scrub RCs
        self.triviback.scrub()
        
        # verify that reference counters are the same as before
        for key in original_db.list():
            if len(key) % 2 == 1 and bytes((key[0], )) == self.triviback.seccs_f._reference_counter.prefix:
                self.assertEqual(original_db.get(key), db.get(key))
        
        # create random contents and check whether they are deleted during scrub
        original_db_len = db.length()
        for _ in range(10, 200):
            random_content = os.urandom(random.randint(0, 102400))
            seccs = [self.triviback.seccs_f, self.triviback.seccs_d, self.triviback.seccs_b][random.randint(0, 2)]
            seccs.put_content(random_content)
        self.assertNotEqual(original_db_len, db.length())
        self.triviback.scrub()
        self.assertEqual(original_db_len, db.length())

        # delete backups
        for (backup_id, backup_path) in backups:
            self.triviback.delete_backup_by_id(backup_id)

if Triviback.get_database_class('semidbm'):
    #@unittest.skip('not now')
    class TrivibackSemiDBMDatabaseTest(AbstractTrivibackClientOnlyTest, unittest.TestCase):
        
        def setUp(self):
            super(TrivibackSemiDBMDatabaseTest, self).setUp()
            self.config_path = self._get_temp_file()
            self.database_path = self._get_temp_directory()
            self.triviback = Triviback(self.config_path, database_type='semidbm', database_conf=self.database_path)
    
        def tearDown(self):
            del self.triviback
            super(TrivibackSemiDBMDatabaseTest, self).tearDown()

if Triviback.get_database_class('zodb'):
    #@unittest.skip('not now')
    class TrivibackZODBDatabaseTest(AbstractTrivibackClientOnlyTest, unittest.TestCase):
        
        def setUp(self):
            super(TrivibackZODBDatabaseTest, self).setUp()
            self.config_path = self._get_temp_file()
            self.database_path = self._get_temp_directory()
            self.triviback = Triviback(self.config_path, database_type='zodb', database_conf=self.database_path)
    
        def tearDown(self):
            del self.triviback
            super(TrivibackZODBDatabaseTest, self).tearDown()

if Triviback.get_database_class('sqlite'):
    #@unittest.skip('not now')
    class TrivibackSQLiteDatabaseTest(AbstractTrivibackClientOnlyTest, unittest.TestCase):
        
        def setUp(self):
            super(TrivibackSQLiteDatabaseTest, self).setUp()
            self.config_path = self._get_temp_file()
            self.database_path = self._get_temp_file()
            os.remove(self.database_path)
            self.triviback = Triviback(self.config_path, database_type='sqlite', database_conf=self.database_path)
    
        def tearDown(self):
            del self.triviback
            super(TrivibackSQLiteDatabaseTest, self).tearDown()

'''
Implementation of test cases for the __init__ server that do not involve synchronization with clients.
'''

#@unittest.skip('not now')
class TrivibackServerTest(unittest.TestCase):
    
    def setUp(self):
        self.triviback_server = server.TrivibackServer()
        app = Flask(__name__)
        self.triviback_server.register_on_app(app)
        self.app = app.test_client()

    def tearDown(self):
        del self.triviback_server
        
    def testSingleElement(self):
        app = self.app
        
        # do the test with randomly chosen content key
        k = os.urandom(16)
        
        # ensure that retrieval of non-existent content issues 404 error
        self.assertEqual(app.get('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii'))).status_code, 404)

        # chose random content
        v = os.urandom(1024)
        
        # insert content
        self.assertEqual(app.put('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii')), data=v).status_code, 200)
        
        # verify retrieval of previously inserted content
        response = app.get('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii')))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, v)

        # overwrite content and verify
        v = os.urandom(1024)
        self.assertEqual(app.put('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii')), data=v).status_code, 200)
        response = app.get('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii')))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, v)

        # delete content and verify
        self.assertEqual(app.delete('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii')), data=v).status_code, 200)
        self.assertEqual(app.get('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii'))).status_code, 404)

    def testMultipleElements(self):
        app = self.app

        # generate random elements        
        kvs = dict()
        for _ in range(10):
            k = os.urandom(16)
            v = os.urandom(1024)
            kvs[k] = v
            
        # send all elements to server at once
        commands = [('put', list(kvs.items()))]
        encoded_commands = msgpack.packb(commands, use_bin_type=True)
        self.assertEqual(app.post('/', data=encoded_commands).status_code, 200)
        
        # request individual elements from server
        for (k, v) in kvs.items():
            response = app.get('/{}'.format(base64.urlsafe_b64encode(k).decode('ascii')))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, v)
        
        # delete some elements, request the remaining elements
        get_list = []
        delete_list = []
        commands = [('get', get_list), ('delete', delete_list)]
        for i, (k, v) in enumerate(sorted(list(kvs.items()))):
            if i % 2 == 0:
                get_list.append(k)
            else:
                delete_list.append(k)
        encoded_commands = msgpack.packb(commands, use_bin_type=True)
        response = app.post('/', data=encoded_commands)
        self.assertEqual(response.status_code, 200)
        results = msgpack.unpackb(response.data)
        for i, (k, v) in enumerate(sorted(list(kvs.items()))):
            if i % 2 == 0:
                self.assertEqual(results.pop(0), v)

'''
Implementation of test cases for __init__ backends.
'''

class AbstractBackendTest():
    
    def __init__(self, backend_type, backend_conf=None):
        self.backend_type = backend_type
        self.backend_conf = backend_conf

    def setUp(self):
        self.triviback = Triviback()
        self.backend = self.triviback.get_backend(self.triviback.add_backend(self.backend_type, self.backend_conf, store_rc=True))
        
    def tearDown(self):
        del self.triviback
        
    def testGetPutDelete(self):
        triviback = self.triviback
        backend = self.backend

        # verify get for random non-existing entry
        k = os.urandom(random.randint(triviback.digest_size, triviback.digest_size+1))
        self.assertEqual(backend.get(k), None)
        
        # verify get for existing entry
        v = os.urandom(random.randint(50, 300))
        backend.put(k, v)
        self.assertEqual(backend.get(k), v)
        
        # verify overwrite of existing entry
        v = os.urandom(random.randint(50, 300))
        backend.put(k, v)
        self.assertEqual(backend.get(k), v)

        # verify delete of existing entry
        backend.delete(k)
        self.assertEqual(backend.get(k), None)

        # verify delete of non-existing entry
        backend.delete(k)
        self.assertEqual(backend.get(k), None)

    def testList(self):
        triviback = self.triviback
        backend = self.backend
        
        # verify list of empty backend
        self.assertEqual(backend.list(), [])
        
        # insert random entries
        d = {}
        for _ in range(100):
            k = os.urandom(random.randint(triviback.digest_size, triviback.digest_size+1))
            v = os.urandom(random.randint(50, 300))
            d[k] = v
            backend.put(k, v)
            
        # verify list
        self.assertEqual(set(backend.list()), set(d.keys()))

    def testPost(self):
        triviback = self.triviback
        backend = self.backend
        
        # verify empty post requests
        self.assertEqual(backend.post([]), [])
        self.assertEqual(backend.post([('get', []), ('put', []), ('delete', []), ('get', [])]), [])
        
        # perform random actions, issue post requests every 100 actions
        d = {}
        post_request = []
        post_expected_response = []
        for i in range(1000):
            # work with an existing or non-existing element each with probability 0.5
            k = random.choice(list(d)) if random.randint(0, 1) == 1 and len(d.keys()) > 0 else os.urandom(random.randint(triviback.digest_size, triviback.digest_size+1))

            # perform random action
            action = random.randint(0, 2)
            if action == 0: # get random element
                # add to post request
                if len(post_request) > 0 and post_request[-1][0] == 'get':
                    post_request[-1][1].append(k)
                else:
                    post_request.append(('get', [k]))
                    
                # remember expected reply
                post_expected_response.append(d[k] if k in d else None)
                
            elif action == 1: # put random element
                # perform action locally
                v = os.urandom(random.randint(50, 300))
                d[k] = v
                
                # add to post request
                if len(post_request) > 0 and post_request[-1][0] == 'put':
                    post_request[-1][1].append((k, v))
                else:
                    post_request.append(('put', [(k, v)]))

            elif action == 2: # delete random element
                # perform action locally
                if k in d: del d[k]
                
                # add to post request
                if len(post_request) > 0 and post_request[-1][0] == 'delete':
                    post_request[-1][1].append(k)
                else:
                    post_request.append(('delete', [k]))
            
            # every 100 actions issue post request and verify response
            if i % 100 == 0:
                self.assertEqual(backend.post(post_request), post_expected_response)
                del post_request[:]
                del post_expected_response[:]

'''
The test cases for the backends should be executed for each available backend implementation.
'''

#@unittest.skip('not now')
class TrivibackMemoryBackendTest(AbstractBackendTest, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        AbstractBackendTest.__init__(self, 'memory')

#@unittest.skip('not now')
class TrivibackServerBackendTest(AbstractBackendTest, unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        
        def get_open_port():
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("",0))
            port = s.getsockname()[1]
            s.close()
            return port
        
        self.host = '127.0.0.1'
        self.port = get_open_port()
        AbstractBackendTest.__init__(self, 'server', backend_conf={'url': 'http://{}:{}/'.format(self.host, self.port)})

    def setUp(self):
        super(TrivibackServerBackendTest, self).setUp()

        self.server_process = Process(target=server.run_server, kwargs={'debug': False, 'host': self.host, 'port': self.port}) # debug=False is important!
        self.server_process.start()
        
        # wait for server to come up
        import time
        time.sleep(5)
        
    def tearDown(self):
        self.server_process.terminate()
        self.server_process.join(1)
        
        super(TrivibackServerBackendTest, self).tearDown()

'''
Implementation of test cases for the synchronization between a __init__ client and backends.
'''

#@unittest.skip('not now')
class TrivibackClientServerTest(AbstractTrivibackTest, unittest.TestCase):

    def setUp(self):
        super(TrivibackClientServerTest, self).setUp()
        self.triviback = Triviback(journal_path=self._get_temp_file())
        
    def tearDown(self):
        del self.triviback
        super(TrivibackClientServerTest, self).tearDown()

    #@unittest.skip('not now')
    def testSingleBackendStandardCase(self):
        triviback = self.triviback
        
        # add a single backend
        backend = triviback.get_backend(triviback.add_backend('memory', None, store_rc=True))
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to backend
                if random.randint(0, 1) == 0:
                    triviback.sync_backends()
                    
                    # local database and backend state must be equal now
                    self.assertEqual(triviback.db.dump(), backend.dump())
                    
                    # journal should be empty now
                    self.assertTrue(triviback.journal.is_empty())

    #@unittest.skip('not now')
    def testTwoBackendsStandardCase(self):
        triviback = self.triviback
        
        # add two backends
        backend1_id = triviback.add_backend('memory', None, store_rc=True)
        backend1 = triviback.get_backend(backend1_id)
        backend2_id = triviback.add_backend('memory', None, store_rc=True)
        backend2 = triviback.get_backend(backend2_id)
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to some backend
                if random.randint(0, 1) == 0:
                    
                    # with probability 1/4, synchronize with backend1, backend2, backend1+backend2 or none of them
                    synchronization_decision = random.randint(0, 3)
                    sync_backend1 = synchronization_decision == 0 or synchronization_decision == 2
                    sync_backend2 = synchronization_decision == 1 or synchronization_decision == 2
                    if sync_backend1 and sync_backend2:
                        triviback.sync_backends()
                    elif sync_backend1:
                        triviback.sync_backends(backend1_id)
                    elif sync_backend2:
                        triviback.sync_backends(backend2_id)
                    
                    # verify states of synced backends
                    if sync_backend1: self.assertEqual(triviback.db.dump(), backend1.dump())
                    if sync_backend2: self.assertEqual(triviback.db.dump(), backend2.dump())
                    
                    # journal should be empty if both backends are synced
                    if sync_backend1 and sync_backend2:
                        self.assertTrue(triviback.journal.is_empty())

    #@unittest.skip('not now')
    def testManyBackends(self):
        triviback = self.triviback

        # start without backends
        backend_ids = []
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/10 add a new backend and sync with it
                backend_ids.append(triviback.add_backend('memory', None, store_rc=True))
                triviback.sync_backends(backend_ids[-1], force=True)
                
                # with probability 1/2 synchronize to some backend if any is available
                if random.randint(0, 1) == 0 and backend_ids:
                    
                    # select backends randomly for synchronization (each with probability 1/3)
                    synchronize_backend_ids = []
                    for backend_id in backend_ids:
                        if random.randint(0, 2) == 0:
                            synchronize_backend_ids.append(backend_id)
                            
                    # do the synchronization
                    if set(synchronize_backend_ids) == set(backend_ids):
                        triviback.sync_backends()
                        
                        # journal should be empty now
                        self.assertTrue(triviback.journal.is_empty())
                        
                    else:
                        for backend_id in synchronize_backend_ids:
                            triviback.sync_backends(backend_id)
                            
                    # verify synchronization
                    for backend_id in synchronize_backend_ids:
                        self.assertEqual(triviback.db.dump(), triviback.get_backend(backend_id).dump())

    #@unittest.skip('not now')
    def testTwoBackendsBrokenJournalCase(self):
        triviback = self.triviback
        
        # add two backends
        backend1_id = triviback.add_backend('memory', None, store_rc=True)
        backend1 = triviback.get_backend(backend1_id)
        backend2_id = triviback.add_backend('memory', None, store_rc=True)
        backend2 = triviback.get_backend(backend2_id)
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/20 destroy journal
                if random.randint(0, 9) == 0:
                    triviback.journal.clear()
                
                # with probability 1/2 synchronize to some backend
                if random.randint(0, 1) == 0:
                    
                    # with probability 1/4, synchronize with backend1, backend2, backend1+backend2 or none of them
                    synchronization_decision = random.randint(0, 3)
                    sync_backend1 = synchronization_decision == 0 or synchronization_decision == 2
                    sync_backend2 = synchronization_decision == 1 or synchronization_decision == 2
                    if sync_backend1 and sync_backend2:
                        triviback.sync_backends(force=True)
                    elif sync_backend1:
                        triviback.sync_backends(backend1_id, force=True)
                    elif sync_backend2:
                        triviback.sync_backends(backend2_id, force=True)
                    
                    # verify states of synced backends
                    if sync_backend1: self.assertEqual(triviback.db.dump(), backend1.dump())
                    if sync_backend2: self.assertEqual(triviback.db.dump(), backend2.dump())
                    
                    # journal should be empty if both backends are synced
                    if sync_backend1 and sync_backend2:
                        self.assertTrue(triviback.journal.is_empty())

    #@unittest.skip('not now')
    def testSingleBackendRecoverCase(self):
        triviback = self.triviback
        
        # add a single backend
        backend = triviback.get_backend(triviback.add_backend('memory', None, store_rc=True))
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to backend
                if random.randint(0, 1) == 0:
                    triviback.sync_backends()
                    
                    # local database and backend state must be equal now
                    self.assertEqual(triviback.db.dump(), backend.dump())
                    
                    # journal should be empty now
                    self.assertTrue(triviback.journal.is_empty())
                    
                    # now delete local database, recover it from backend, and verify
                    triviback.db._dict.clear()
                    triviback._backups = None
                    triviback._state = None
                    triviback._state_content_key = None
                    triviback.sync_backends(force=True)
                    self.assertEqual(backend.dump(), triviback.db.dump())

    #@unittest.skip('not now')
    def testTwoBackendsRecoverCase(self):
        triviback = self.triviback
        
        # add two backends
        backend1_id = triviback.add_backend('memory', None, store_rc=True)
        backend1 = triviback.get_backend(backend1_id)
        backend2_id = triviback.add_backend('memory', None, store_rc=True)
        backend2 = triviback.get_backend(backend2_id)
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to some backend
                if random.randint(0, 1) == 0:
                    
                    # with probability 1/4, synchronize with backend1, backend2, backend1+backend2 or none of them
                    synchronization_decision = random.randint(0, 3)
                    sync_backend1 = synchronization_decision == 0 or synchronization_decision == 2
                    sync_backend2 = synchronization_decision == 1 or synchronization_decision == 2
                    if sync_backend1 and sync_backend2:
                        triviback.sync_backends()
                    elif sync_backend1:
                        triviback.sync_backends(backend1_id)
                    elif sync_backend2:
                        triviback.sync_backends(backend2_id)
                    
                    # verify states of synced backends
                    if sync_backend1: self.assertEqual(triviback.db.dump(), backend1.dump())
                    if sync_backend2: self.assertEqual(triviback.db.dump(), backend2.dump())
                    
                    # journal should be empty if both backends are synced
                    if sync_backend1 and sync_backend2:
                        self.assertTrue(triviback.journal.is_empty())

                    # if at least one backend is in sync, delete local database, recover it from backend, and verify
                    if sync_backend1 or sync_backend2:
                        triviback.db._dict.clear()
                        triviback._backups = None
                        triviback._state = None
                        triviback._state_content_key = None
                        triviback.sync_backends(force=True)
                        
                        if sync_backend1: self.assertEqual(backend1.dump(), triviback.db.dump())
                        if sync_backend2: self.assertEqual(backend2.dump(), triviback.db.dump())

    #@unittest.skip('not now')
    def testTwoBackendsRemoteImmutableDataCorruptionRecoverCase(self):
        triviback = self.triviback
        
        # add two backends
        backend1_id = triviback.add_backend('memory', None, store_rc=True)
        backend1 = triviback.get_backend(backend1_id)
        backend2_id = triviback.add_backend('memory', None, store_rc=True)
        backend2 = triviback.get_backend(backend2_id)
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to some backend
                if random.randint(0, 1) == 0:
                    
                    # with probability 1/4, synchronize with backend1, backend2, backend1+backend2 or none of them
                    synchronization_decision = random.randint(0, 3)
                    sync_backend1 = synchronization_decision == 0 or synchronization_decision == 2
                    sync_backend2 = synchronization_decision == 1 or synchronization_decision == 2
                    if sync_backend1 and sync_backend2:
                        triviback.sync_backends()
                    elif sync_backend1:
                        triviback.sync_backends(backend1_id)
                    elif sync_backend2:
                        triviback.sync_backends(backend2_id)
                    
                    # verify states of synced backends
                    if sync_backend1: self.assertEqual(triviback.db.dump(), backend1.dump())
                    if sync_backend2: self.assertEqual(triviback.db.dump(), backend2.dump())
                    
                    # journal should be empty if both backends are synced
                    if sync_backend1 and sync_backend2:
                        self.assertTrue(triviback.journal.is_empty())

                    # test recovery only if both backends are in sync
                    if sync_backend1 and sync_backend2:
                        # chose one backend on which immutable entries are corrupted at random
                        corrupt_backend = backend1 if random.randint(0, 1) == 0 else backend2
                        ks = [k for k in triviback.db._dict.keys() if len(k) == triviback.digest_size]
                        random.shuffle(ks)
                        for k in ks[:100]: # corrupt at most 100 entries
                            if random.randint(0, 4) == 0:
                                corrupt_backend.delete(k)
                            else:
                                corrupt_backend.put(k, os.urandom(random.randint(50, 300)))
                        
                        # clear local data base
                        triviback.db._dict.clear()
                        triviback._backups = None
                        triviback._state = None
                        triviback._state_content_key = None
                        
                        # recover uncorrrupted data from backends
                        triviback.sync_backends(force=True)
                        
                        # synchronize uncorrupted state back to backends
                        backend1.clear()
                        backend2.clear()
                        triviback.sync_backends(force=True)
                        
        # verify recovery of all backups
        for backup in triviback.backups:
            backup_id = backup[0]
            recover_path = self._get_temp_directory()
            triviback.recover_path_by_id(backup_id, recover_path)

    #@unittest.skip('not now')
    def testTwoBackendsRemoteMutableDataCorruptionRecoverCase(self):
        triviback = self.triviback
        
        # add two backends
        backend1_id = triviback.add_backend('memory', None, store_rc=True)
        backend1 = triviback.get_backend(backend1_id)
        backend2_id = triviback.add_backend('memory', None, store_rc=True)
        backend2 = triviback.get_backend(backend2_id)
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to some backend
                if random.randint(0, 1) == 0:
                    
                    # with probability 1/4, synchronize with backend1, backend2, backend1+backend2 or none of them
                    synchronization_decision = random.randint(0, 3)
                    sync_backend1 = synchronization_decision == 0 or synchronization_decision == 2
                    sync_backend2 = synchronization_decision == 1 or synchronization_decision == 2
                    if sync_backend1 and sync_backend2:
                        triviback.sync_backends()
                    elif sync_backend1:
                        triviback.sync_backends(backend1_id)
                    elif sync_backend2:
                        triviback.sync_backends(backend2_id)
                    
                    # verify states of synced backends
                    if sync_backend1: self.assertEqual(triviback.db.dump(), backend1.dump())
                    if sync_backend2: self.assertEqual(triviback.db.dump(), backend2.dump())
                    
                    # journal should be empty if both backends are synced
                    if sync_backend1 and sync_backend2:
                        self.assertTrue(triviback.journal.is_empty())

                    # test recovery only if both backends are in sync
                    if sync_backend1 and sync_backend2:
                        # chose one backend on which immutable entries are corrupted at random
                        corrupt_backend = backend1 if random.randint(0, 1) == 0 else backend2
                        ks = [k for k in triviback.db._dict.keys() if len(k) != triviback.digest_size]
                        random.shuffle(ks)
                        for k in ks[:100]: # corrupt at most 100 entries
                            if random.randint(0, 4) == 0:
                                corrupt_backend.delete(k)
                            else:
                                # RCs are only decreased, other mutable entries are modified at random
                                try:
                                    corrupt_backend.put(k, struct.pack('!Q', struct.unpack('!Q', corrupt_backend.get(k) or '\x00\x00\x00\x00\x00\x00\x00\x00')[0] - 1))
                                except:
                                    corrupt_backend.put(k, os.urandom(random.randint(50, 300)))
                        
                        # clear local data base
                        triviback.db._dict.clear()
                        triviback._backups = None
                        triviback._state = None
                        triviback._state_content_key = None
                        
                        # recover uncorrupted data from backends
                        triviback.sync_backends(force=True)
                        
                        # synchronize uncorrupted state back to backends
                        backend1.clear()
                        backend2.clear()
                        triviback.sync_backends(force=True)
                        
        # verify recovery of all backups
        for backup in triviback.backups:
            backup_id = backup[0]
            recover_path = self._get_temp_directory()
            triviback.recover_path_by_id(backup_id, recover_path)

    #@unittest.skip('not now')
    def testManyBackendsRemoteMutableDataCorruptionRecoverCase(self):
        triviback = self.triviback

        # start with three backends
        backend_ids = [triviback.add_backend('memory', None, store_rc=True), triviback.add_backend('memory', None, store_rc=True), triviback.add_backend('memory', None, store_rc=True)]
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)

                # with probability 1/10 add a new backend
                new_backend_id = triviback.add_backend('memory', None, store_rc=True)
                backend_ids.append(new_backend_id)
                triviback.sync_backends(new_backend_id, force=True)
                
                # with probability 1/2 synchronize to some backend
                if random.randint(0, 1) == 0:
                    
                    # select backends randomly for synchronization (each with probability 1/3)
                    synchronize_backend_ids = []
                    for backend_id in backend_ids:
                        if random.randint(0, 2) == 0:
                            synchronize_backend_ids.append(backend_id)
                            
                    # do the synchronization
                    if set(synchronize_backend_ids) == set(backend_ids):
                        triviback.sync_backends()
                        
                        # journal should be empty now
                        self.assertTrue(triviback.journal.is_empty())
                        
                    else:
                        for backend_id in synchronize_backend_ids:
                            triviback.sync_backends(backend_id)
                    
                    # verify synchronization
                    for backend_id in synchronize_backend_ids:
                        self.assertEqual(triviback.db.dump(), triviback.get_backend(backend_id).dump())

                    # test recovery only if at least three backends are in sync
                    if len(synchronize_backend_ids) >= 3:
                        
                        # corrupt less than half of the backends
                        random.shuffle(synchronize_backend_ids)
                        corrupted_backend_ids = synchronize_backend_ids[:int((len(synchronize_backend_ids)-1)/2)]
                        assert len(corrupted_backend_ids) < len(synchronize_backend_ids)/2.0
                        
                        # corrupt arbitrary entries
                        for corrupt_backend_id in corrupted_backend_ids:
                            corrupt_backend = triviback.get_backend(corrupt_backend_id)
                            ks = [k for k in triviback.db._dict.keys()]
                            random.shuffle(ks)
                            for k in ks[:100]: # corrupt at most 100 entries
                                if random.randint(0, 4) == 0:
                                    corrupt_backend.delete(k)
                                else:
                                    corrupt_backend.put(k, os.urandom(random.randint(50, 300)))
                            # also add some random entries
                            for _ in range(50):
                                k = os.urandom(random.randint(triviback.digest_size, triviback.digest_size+1))
                                corrupt_backend.put(k, os.urandom(random.randint(50, 300)))
                        
                        # clear local data base
                        triviback.db._dict.clear()
                        triviback._backups = None
                        triviback._state = None
                        triviback._state_content_key = None
                        
                        # recover (hopefully uncorrupted) data from backends
                        triviback.sync_backends(force=True)
                        
                        # verify that local state matches the state of some uncorrupted backend
                        self.assertEqual(triviback.db._dict, triviback.get_backend(list(set(synchronize_backend_ids) - set(corrupted_backend_ids))[0]).dump())
                        
                        # synchronize uncorrupted state back to backends
                        for backend_id in backend_ids:
                            triviback.get_backend(backend_id).clear()
                        triviback.sync_backends(force=True)
                        
        # verify recovery of all backups
        for backup in triviback.backups:
            backup_id = backup[0]
            recover_path = self._get_temp_directory()
            triviback.recover_path_by_id(backup_id, recover_path)

    #@unittest.skip('not now')
    def testSingleBackendRecoverLocalInconsistencyCase(self):
        triviback = self.triviback
        
        # add a single backend
        backend = triviback.get_backend(triviback.add_backend('memory', None, store_rc=True))
        
        # create directory that is backed up
        source_path = self._get_temp_directory()

        # perform random actions
        workload_generator = AbstractTrivibackTest.WorkloadGenerator(source_path)
        for _ in range(0, 1000):
            workload_generator.random_action()
            
            # with probability 1/10 create a new backup
            if random.randint(0, 9) == 0:
                backup_path = self._get_temp_directory()
                shutil.rmtree(backup_path)
                shutil.copytree(source_path, backup_path)
                triviback.backup_path(source_path)
                
                # with probability 1/2 synchronize to backend
                if random.randint(0, 1) == 0:
                    triviback.sync_backends()
                    
                    # local database and backend state must be equal now
                    self.assertEqual(triviback.db.dump(), backend.dump())
                    
                    # journal should be empty now
                    self.assertTrue(triviback.journal.is_empty())
                    
                    # now manipulate _all_ mutable and some immutable items in local database
                    for k in list(triviback.db._dict.keys()):
                        if random.randint(0, 4) == 0:
                            triviback.db.delete(k)
                        else:
                            # only modify mutable entries since immutable entries are skipped during synchronization
                            if len(k) != triviback.digest_size: triviback.db.put(k, os.urandom(random.randint(50, 300)))
                    for _ in range(50):
                        triviback.db.put(os.urandom(random.randint(triviback.digest_size, triviback.digest_size+1)), os.urandom(random.randint(50, 300)))
                    
                    # recover local database and verify
                    triviback._backups = None
                    triviback._state = None
                    triviback._state_content_key = None
                    triviback.sync_backends(force=True)
                    self.assertEqual(backend.dump(), triviback.db.dump())


'''
Run the tests.
'''

def run_tests():
    unittest.main(failfast=True)

if __name__ == "__main__":
    run_tests()
