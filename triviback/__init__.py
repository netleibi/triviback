from _collections import defaultdict
from _functools import reduce
import argparse
import base64
import codecs
from collections import Counter
import datetime
from operator import concat
import os.path
import struct
import time
import zlib
import logging

import argh
from argh.decorators import arg
from argh.exceptions import CommandError
import msgpack
from tabulate import tabulate
import yaml

from triviback.backend import AbstractBackend
from triviback.database import AbstractDatabase


try:
    import fastchunking
except ImportError:
    from pkg_resources import require
    require("fastchunking")
    import fastchunking
    
try:
    import seccs
except ImportError:
    from pkg_resources import require
    require("seccs")
    import seccs

logger = logging.getLogger(__name__)

__version__ = '0.0.1'


class TrivibackJournal():
    
    def __init__(self, journal_path=None):
        if journal_path is not None:
            journal_dir = os.path.dirname(journal_path)
            if journal_dir and not os.path.exists(journal_dir):
                os.makedirs(journal_dir)
        self.journal_file = journal_file = codecs.open(journal_path, 'r+' if os.path.exists(journal_path) else 'w+', encoding='ascii')
        
        # seek to end of journal (which is defined by the last commit)
        journal_offset = current_offset = 0
        for line in journal_file:
            current_offset += len(line)
            if line.startswith('V'):
                journal_offset = current_offset
        journal_file.seek(journal_offset)
        journal_file.truncate()
        
        self._transaction_active = False
        self._buffer = []
        
    def journalize_fn(self, fn, op_code, k_argument_number=0):
        journal_file = self.journal_file
        def wrapped_fn(*args, **kwargs):
            result = fn(*args, **kwargs)
            journal_file.write('{}{}\n'.format(op_code, base64.b64encode(args[k_argument_number]).decode('ascii')))
            return result
        return wrapped_fn
    
    def transaction_start(self):
        self._transaction_active = True
    
    def transaction_commit(self, filter_ops=[]):
        journal_file = self.journal_file
        buffer = self._buffer
        while buffer:
            op_code, data = buffer.pop(0)
            if op_code not in filter_ops:
                journal_file.write('{}{}\n'.format(op_code, data).decode('ascii'))
        self._transaction_active = False
    
    def close(self):
        self.journal_file.close()
    
    def commit(self, version):
        self.journal_file.write('V{}\n'.format(version))
        self.flush()
        
    def clear(self):
        self.journal_file.truncate(0)
        self.journal_file.seek(0)

    def flush(self):
        self.journal_file.flush()
        
    def is_empty(self):
        return self.journal_file.tell() == 0
        
    def get_changes(self, base_version, target_version):
        # read all changes between the two specified versions from journal file
        journal_file = self.journal_file
        journal_file.seek(0)
        changes = defaultdict(list)
        reached_base_version = 0
        changes_since_base_version_available = False
        for line in journal_file:
            # detect version change
            if line[0] == 'V':
                reached_base_version = int(line[1:])
                
                # if we see a commit of a version lower or equal to the base version,
                # all changes seen so far are irrelevant
                if reached_base_version <= base_version:
                    changes.clear()
                    
                # to provide consistency, the journal MUST contain base_version + 1
                if reached_base_version == base_version + 1:
                    changes_since_base_version_available = True
                    
                continue
            
            # stop when we are beyond the changes we were interested in
            if reached_base_version >= target_version:
                break

            # read change
            op_code, k = line[0], base64.b64decode(line[1:])
            changes[k].append(op_code)
        
        # ensure that all changes from base_version to target_version were actually in the journal
        if reached_base_version < target_version or not changes_since_base_version_available:
            return None
        
        # sort and compact changes by filtering out redundant ones
        put_changes = []
        delete_changes = []
        for k, operations in sorted(changes.items()):
            inc_count, dec_count = operations.count('+'), operations.count('-')
            if (inc_count > 0 or dec_count > 0) and inc_count == dec_count: # if RC item is unchanged, we do not have to do anything
                pass
            else: # for non-RC entries and changed RC entries, only the most-recent operation is interesting
                last_operation = reduce(lambda accum, op: accum if op == '+' or op == '-' else op, operations)
                if last_operation == 'P':
                    put_changes.append(k)
                elif last_operation == 'D':
                    delete_changes.append(k)
                else:
                    raise Exception('Unsupported operation: {}'.format(last_operation))

        # return compacted list of changes        
        return [('put', put_changes), ('delete', delete_changes)]

def dict_to_list(obj):
    # convert dict to list
    if isinstance(obj, dict):
        return sorted([(k, dict_to_list(v)) for (k, v) in obj.items()])
    return obj

def list_to_dict(obj):
    # convert list to dict
    if isinstance(obj, list):
        return dict([(k, list_to_dict(v)) for (k, v) in obj])
    return obj

class Triviback():
    '''
    Triviback is a simple, but storage-efficient, authenticated and encrypted backup system based on the sec-cs data structure.
    
    An instance of the backup system consists of:
    * a secret(!) CONFIGURATION file which is initialized during set-up
    * a JOURNAL file that tracks data structure changes between backup versions
    * a local DATABASE storing the actual backups
    * an arbitrary number of optional BACKENDS that allow to recover the database
    
    After the backup is set up including the backends, the user has to store the configuration file at some secure place.
    The configuration file alone is sufficient to restore any further backups from the backends.
    
    Local and backend databases are organized as key-value stores. The following convention holds for its elements:
    * Contents managed by sec-cs are stored under keys of length digest_size,
      digest sizes must be even, i.e., all values stored under keys of even length are IMMUTABLE.
    * Any other contents are stored under keys of odd length,
      i.e., all values stored under keys of odd length are considered MUTABLE.
    This differentiation is important as the synchronization function between local database and backends depends on it.
    '''
    
    def __init__(self, config_path=None, journal_path=None, database_conf=None, database_type=None, chunk_size=None, window_size=None, grow_forever=None, use_compression=None):
        # load or initialize configuration
        self.config_path = config_path
        if config_path is not None:
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
        self.config = yaml.safe_load(codecs.open(config_path, 'r', encoding='utf8')) if config_path is not None and os.path.isfile(config_path) else None
        if self.config is None: self.config = {}
        config_changed = False

        # journal configuration
        try:
            journal_path = self.config['journal_path']
        except KeyError:
            # default to creating journal in config file path if it exists
            if not journal_path and config_path: journal_path = os.path.join(os.path.dirname(config_path), 'triviback.journal')
            # use absolute journal_path unless it is in a subdirectory of the config file
            if journal_path and not os.path.abspath(journal_path).startswith(os.path.join(os.path.dirname(os.path.abspath(config_path)), '')):
                journal_path = os.path.abspath(journal_path)
            self.config['journal_path'] = journal_path
            config_changed = True
        self.journal = journal = TrivibackJournal(journal_path) if journal_path is not None else None

        # database configuration
        try:
            try:
                database = self.config['database']
            except KeyError:
                database = {}
                self.config['database'] = database
                raise
            database_conf = database['conf']
            database_type = database['type']
        except KeyError:
            if database_type is None: database_type = 'memory'
            
            # default to creating database in config file path
            if not database_conf and config_path: database_conf = os.path.join(os.path.dirname(config_path), 'triviback.db')
            # use absolute database path unless it is in a subdirectory of the config file
            if database_conf and not os.path.abspath(database_conf).startswith(os.path.join(os.path.dirname(os.path.abspath(config_path)), '')):
                database_conf = os.path.abspath(database_conf)

            database['conf'] = database_conf
            database['type'] = database_type
            config_changed = True

        # load / initialize database
        db_cls = self.get_database_class(database_type)
        if db_cls is None:
            raise Exception('Unsupported database type: {}'.format(database_type))
        self.db = db = db_cls(database_conf)
        
        # extract required functions from database
        put_fn = db.put
        get_fn = db.get
        delete_fn = db.delete
        list_fn = db.list

        # journalize functions
        if journal is not None:
            journalized_put_fn = self.journal.journalize_fn(put_fn, 'P')
            journalized_delete_fn = self.journal.journalize_fn(delete_fn, 'D')
            
            put_fn = journalized_put_fn
            delete_fn = journalized_delete_fn

        # sec-cs configuration
        try:
            try:
                seccs_config = self.config['seccs']
            except KeyError:
                seccs_config = {}
                self.config['seccs'] = seccs_config
                raise
            window_size = int(seccs_config['window_size'])
            chunk_size = int(seccs_config['chunk_size'])
            grow_forever = bool(seccs_config['grow_forever'])
            use_compression = bool(seccs_config['use_compression'])
        except KeyError:
            if window_size is None: window_size = 48
            if chunk_size is None: chunk_size = 256
            if grow_forever is None: grow_forever = False
            if use_compression is None: use_compression = False
            seccs_config['window_size'] = window_size
            seccs_config['chunk_size'] = chunk_size
            seccs_config['grow_forever'] = grow_forever
            seccs_config['use_compression'] = use_compression
            config_changed = True

        # enable / disable compression
        if not use_compression:
            self._maybe_compress = lambda x: x
            self._maybe_decompress = lambda x: x

        # use RabinKarp rolling hash for chunking
        rolling_hash = fastchunking.RabinKarpCDC(window_size, 0) # FIXME: use random seed instead of 0

        # determine/choose encryption keys
        try:
            encryption_keys = self.config['encryption_keys']
        except KeyError:
            self.config['encryption_keys'] = encryption_keys = {}
        
        try:
            files_key = encryption_keys['files']
        except KeyError:
            encryption_keys['files'] = files_key = os.urandom(32)
            config_changed = True
        try:
            directories_key = encryption_keys['directories']
        except KeyError:
            encryption_keys['directories'] = directories_key = os.urandom(32)
            config_changed = True
        try:
            backups_key = encryption_keys['backups']
        except KeyError:
            encryption_keys['backups'] = backups_key = os.urandom(32)
            config_changed = True

        # remember parameters for internal use
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.put_fn, self.get_fn, self.delete_fn = put_fn, get_fn, delete_fn
        self.rolling_hash = rolling_hash
        self.grow_forever = grow_forever

        # initialize sec-cs-compatible storage data structure
        dbwrapper = Triviback.get_db_wrapper(put_fn, get_fn, delete_fn, list_fn)

        # enable reference counting unless grow_forever is enabled
        if grow_forever:
            rcwrapper = Triviback.get_rc_wrapper(None, None, None, None)
        else:
            rcwrapper = Triviback.get_rc_wrapper(put_fn, get_fn, delete_fn, list_fn, journal=journal)
        
        self.seccs_f = seccs.SecCSLite(chunk_size, dbwrapper, seccs.crypto_wrapper.AES_SIV_256_DISTINGUISHED_ROOT(files_key), rolling_hash, rcwrapper)
        self.seccs_d = seccs.SecCSLite(chunk_size, dbwrapper, seccs.crypto_wrapper.AES_SIV_256_DISTINGUISHED_ROOT(directories_key), rolling_hash, rcwrapper)
        self.seccs_b = seccs.SecCSLite(chunk_size, dbwrapper, seccs.crypto_wrapper.AES_SIV_256_DISTINGUISHED_ROOT(backups_key), rolling_hash, rcwrapper)
        assert self.seccs_f._crypto_wrapper.DIGEST_SIZE == self.seccs_d._crypto_wrapper.DIGEST_SIZE == self.seccs_b._crypto_wrapper.DIGEST_SIZE
        self.digest_size = digest_size = self.seccs_f._crypto_wrapper.DIGEST_SIZE

        # root key configuration
        try:
            self.root_key = self.config['root_key']
        except KeyError:
            self.config['root_key'] = self.root_key = b'm'+os.urandom(digest_size)  # mutable, thus odd length
            config_changed = True

        # backends configuration
        try:
            backends = self.config['backends']
        except KeyError:
            self.config['backends'] = backends = []
            config_changed = True
        
        # load / initialize backends
        self.backends = {}
        for backend in backends:
            self._do_add_backend(backend['id'], backend['type'], backend['conf'], store_rc=backend.get('store_rc', False))

        # save config
        if config_changed: self.save_config()
        
        # determine current state
        self._state_content_key = None
        self._state = None
        self._backups = None

    def __del__(self):
        # close database
        try:
            self.db
        except AttributeError:
            pass
        else:
            if self.db is not None:
                self.db.close()
        try:
            self.journal
        except AttributeError:
            pass
        else:
            if self.journal is not None:
                self.journal.close()

    def save_config(self):
        # store configuration
        if self.config_path is not None:
            yaml.safe_dump(self.config, codecs.open(self.config_path, 'w', encoding='utf8'), default_flow_style=False)

    '''
    Public interface.
    '''

    def backup_path(self, path, detect_changes=True, progress_fn=None):
        # if allowed and possible, backup only changes to last backup
        try:
            _, base_root_key = max([(ex_timestamp, ex_root_key) for (_, ex_timestamp, ex_path, ex_root_key) in self.backups if ex_path == path]) if detect_changes else (None, None)
        except ValueError:
            base_root_key = None

        # do recursive backup of path
        abs_path = os.path.abspath(path)
        backup_id = max(map(lambda backup: backup[0], self.backups))+1 if len(self.backups) > 0 else 1
        timestamp = time.time()
        self._progress_current_bytes = 0
        self._progress_total_bytes = self._get_total_size_of_path(abs_path)
        backup_root_key = self._do_backup_path(abs_path, base_root_key, progress_fn=progress_fn)
        self.seccs_d._reference_counter.inc(backup_root_key[:self.digest_size])
        self.backups.append((backup_id, timestamp, abs_path, backup_root_key))
        
        # update list of existing backups
        self._persist_state()
        
        return backup_id

    def delete_backup_by_id(self, backup_id):
        # remove at most one backup
        for index, (existing_id, _, _, existing_backup_root_key) in enumerate(self.backups):
            if existing_id == backup_id:
                if self.seccs_d._reference_counter.dec(existing_backup_root_key[:self.digest_size]) == 0:
                    self._do_delete_backup(existing_backup_root_key)
                self.backups[:] = self.backups[:index] + self.backups[index+1:]
                break
        else:
            # raise exception if backup_id did not exist
            raise Exception('Backup not found.')

        # update list of existing backups
        self._persist_state()
        
    def recover_path_by_id(self, backup_id, path, force=False, progress_fn=None):
        # recover existing backup
        for (existing_id, _, _, existing_backup_root_key) in self.backups:
            if existing_id == backup_id:
                self._progress_current_bytes = self._progress_total_bytes = 0
                return self._do_recover_path(existing_backup_root_key, path, force, progress_fn=progress_fn)
        else:
            raise Exception('Backup does not exist.')

    def scrub(self):
        ''' Re-calculate reference counters and remove stale objects. '''
        
        # remember actual reference counters first
        assert self.seccs_f._reference_counter == self.seccs_d._reference_counter == self.seccs_b._reference_counter  # we only consider the case that all seccs instances share an RC
        original_rc = self.seccs_f._reference_counter
        
        # for the purpose of scrub, we temporarily switch to a new, empty reference counter whose
        # values will be computed during this operation
        temporary_rc = self.get_memory_rc_wrapper(original_rc.prefix)  # we only use a single temporary RC as RCs are shared by all seccs instances anyway
        try:
            self.seccs_f._reference_counter = temporary_rc
            self.seccs_d._reference_counter = temporary_rc
            self.seccs_b._reference_counter = temporary_rc
            
            # TODO: introduce more efficient functions for re-calculating RCs in sec-cs.
            # For now, we just re-insert each content once, which yields the same result
            # at the expense of much higher computation time.
            def update_rcs_for_content(seccs, content_key):
                seccs.put_content(seccs.get_content(content_key))
            
            # update RCs for state content and backup list content
            update_rcs_for_content(self.seccs_b, self.state_content_key)
            update_rcs_for_content(self.seccs_b, self.state['backups_key'])
            
            # updating RCs of directory trees requires recursion
            def update_rcs_for_directory(base_key):
                # update RCs for directory object
                update_rcs_for_content(self.seccs_d, base_key)
                
                # update RCs for directory's children
                existing_entries = list_to_dict(msgpack.unpackb(self._maybe_decompress(self.seccs_d.get_content(base_key)), encoding='utf-8'))
                for _, entry in existing_entries.items():
                    child_key = entry['key']
                    if temporary_rc.get(child_key[:self.digest_size]) == 0:
                        if entry['is_file']:
                            # update RCs for file object
                            update_rcs_for_content(self.seccs_f, child_key)
                        else:
                            # recurse
                            update_rcs_for_directory(child_key)
                    else:
                        # children's RCs must be incremented in any case
                        temporary_rc.inc(child_key[:self.digest_size])
            
            # update RCs for backups
            for (_, _, _, backup_key) in self.backups:
                # check if directory with backup_key has an RC already
                if temporary_rc.get(backup_key[:self.digest_size]) == 0:
                    update_rcs_for_directory(backup_key)
                else:
                    temporary_rc.inc(backup_key[:self.digest_size])
        
        finally:
            # switch back to original reference counters
            self.seccs_f._reference_counter = original_rc
            self.seccs_d._reference_counter = original_rc
            self.seccs_b._reference_counter = original_rc
        
        if not self.grow_forever:
            # update RCs with newly computed ones
            updated_rcs = set([])
            for k in temporary_rc:
                original_rc.set(k, temporary_rc.get(k))
                updated_rcs.add(k)
            
            # delete obsolete RCs
            obsolete_rcs = set([])
            for k in original_rc:
                if k not in updated_rcs:
                    obsolete_rcs.add(k)
            for k in obsolete_rcs:
                if original_rc.get(k):
                    original_rc.set(k, 0)
        
        # delete stale objects based on temporary_rc
        assert self.seccs_f._database == self.seccs_d._database == self.seccs_b._database  # we only consider the case that all seccs instances share a database
        database = self.seccs_f._database
        stale_keys = set([])
        for key in database:
            if len(key) % 2 == 0 and temporary_rc.get(key) == 0:
                # immutable (even-length) objects without RC are considered stale
                stale_keys.add(key)
        for stale_key in stale_keys:
            del database[stale_key]

    def add_backend(self, backend_type, backend_conf, store_rc=False):
        if store_rc and self.grow_forever:
            raise Exception('Cannot store RCs on backends if grow_forever is enabled.')
        
        # choose backend ID
        backend_id = 0
        for backend in self.config['backends']:
            backend_id = max(backend_id, int(backend['id']))
        backend_id += 1

        # add backend to configuration
        self.config['backends'].append({'id': backend_id, 'type': backend_type, 'conf': backend_conf, 'store_rc': store_rc})
        
        # load backend and save config
        self._do_add_backend(backend_id, backend_type, backend_conf, store_rc=store_rc)
        self.save_config()
        
        return backend_id
    
    def get_backend(self, backend_id):
        # return backend
        return self.backends[backend_id]

    def delete_backend(self, backend_id):
        # remove first backend with corresponding ID from backends list
        backends = self.config['backends']
        for i, backend in enumerate(backends):
            if backend['id'] == backend_id:
                backends[:] = backends[:i] + backends[i+1:]
                del self.backends[backend['id']]
                break
        else:
            # if no backend has been removed, raise error
            raise Exception('Backend does not exist')

        # save changed configuration
        self.save_config()
        
    def list_backends(self):
        return self.config['backends']
    
    def sync_backends(self, backend_ids=None, force=False):
        # first connect to all desired backends and determine their maximum version
        maximum_backend_version = 0
        backends = {}
        backend_states = {}
        backend_state_content_keys = {}
        for backend_id in (self.backends.keys() if backend_ids is None else [int(backend_id) for backend_id in backend_ids] if isinstance(backend_ids, list) else [int(backend_ids)]):
            # get backend
            backend = self.backends[backend_id]
            backends[backend_id] = backend
            
            # get backend state
            backend_state_content_key = backend.get(self.root_key)
            backend_state_content_keys[backend_id] = backend_state_content_key
            backend_state = {'backups_key': None, 'version': 0}
            if backend_state_content_key is not None:
                try:
                    backend_state = list_to_dict(msgpack.unpackb(self._maybe_decompress(backend.seccs_b.get_content(backend_state_content_key)), encoding='utf-8'))
                except:
                    logger.warning("Failed to fetch backup state object from backend {}".format(backend_id))
                    pass
            else:
                logger.warning("There is no backend state object on backend {}".format(backend_id))
            backend_states[backend_id] = backend_state

            # determine maximum version
            maximum_backend_version = max(maximum_backend_version, backend_state['version'])
            
        # update local state if any backend has a newer state
        if maximum_backend_version > self.state['version']:
            # this should only happen if the user agrees
            if not force: raise Exception('Out of sync: Backend {} has newer state than local database'.format(backend_id))
            
            # determine backends from which data can be retrieved
            appropriate_backend_ids = list(filter(lambda backend_id: backend_states[backend_id]['version'] == maximum_backend_version, backends.keys()))

            # ensure that all backends with identical versions agree on the backup state
            # (due to authentication of backup states, this is the case if their backup state content keys
            # are identical)
            # NOTE that conflicts can only occur if triviback has been improperly used (i.e. different
            # backups with identical versions have been created), so manual conflict resolution is appropriate!
            appropriate_backend_state_content_keys = set([backend_state_content_keys[backend_id] for backend_id in appropriate_backend_ids])
            if len(appropriate_backend_state_content_keys) != 1:
                raise Exception('An inconsistency occurred. Different backends contain different backup states with identical versions.')
            appropriate_backend_state_content_key = list(appropriate_backend_state_content_keys)[0]
            
            # determine which entries are available at the backends
            available_entries = reduce(concat, [list(set(backends[backend_id].list())) for backend_id in appropriate_backend_ids], [])
            
            # filter out entries that are stored by less than half of the backends
            appropriate_backends_count = len(appropriate_backend_ids)
            available_entries = set([k for (k, k_count) in Counter(available_entries).items() if not (k_count < appropriate_backends_count / 2.0)])

            # determine which local entries have to be changed
            local_keys = set(self.db.list())
            changed_immutable_entries = set([k for k in available_entries if len(k) % 2 == 0]) - local_keys
            changed_mutable_entries = set([k for k in available_entries if len(k) % 2 == 1 and k != self.root_key])
            vanished_entries = local_keys - available_entries
            
            # determine whether all backends store RCs
            rc_everywhere = True
            
            # synchronize immutable entries from backends to local database by simply copying the ones that are authentic
            for backend_id in appropriate_backend_ids:
                # stop as soon as there are no immutable entries left
                if not changed_immutable_entries:
                    break

                # retrieve entries from backend
                backend = backends[backend_id]
                entries = zip(changed_immutable_entries, backend.post([('get', changed_immutable_entries)]))
                
                # determine whether all backends store RCs
                rc_everywhere = rc_everywhere and backend.store_rc
                
                # put authentic entries to local database
                for k, v in entries:
                    try:
                        if v is not None:
                            valid = False
                            # FIXME: This is a dirty brute-force workaround which is required since we need context information
                            #   in order to check a node's authenticity since the latest conceptual change. (Before this change,
                            #   authentication of chunk tree node's had been performed irrespective of their positions in chunk
                            #   trees, so authenticity could be checked in a straightforward way (without all the for loops).)
                            # For the future, it might be more suitable to scan the backup/directory hierarchy under the backend's
                            # root node and import all nodes that are not locally available.
                            # This, however, would leak some access patterns. To avoid that, we could just download all immutable
                            # contents from backends and then run a local scrub to get rid of garbage.
                            for is_root in [True, False]:
                                for height in range(20):
                                    for seccs in [self.seccs_f, self.seccs_d, self.seccs_b]:
                                        if not valid:
                                            try:
                                                seccs._crypto_wrapper.unwrap_value(v, k, height, is_root)
                                                valid = True
                                            except ValueError:
                                                pass
                            if valid:
                                self.db.put(k, v)
                    except ValueError:
                        # ignore non-authentic entries
                        pass
                        
                # remove entries which have successfully been retrieved from list
                local_keys = set(self.db.list())
                changed_immutable_entries -= local_keys

            if rc_everywhere:
                # synchronize mutable entries from backends to local database
                #
                # due to lack of authentication of immutable entries, they are retrieved from _all_ backends
                # and synchronized as follows:
                # * if more than half of all backends agree on a value, that value is copied to the local database (majority vote)
                # * otherwise, the maximum value of all backends is copied to the local database
                #   (this ensures that inconsistencies in single backends cannot cause too low reference counter
                #   values which might lead to data loss)
                # TODO: notify the user that performing a scrub might be a good idea...
                entries = zip(changed_mutable_entries, zip(*[backends[backend_id].post([('get', changed_mutable_entries)]) for backend_id in appropriate_backend_ids]))
                for k, v_list in entries:
                    majority_count, majority_v = sorted([(v_list.count(v), v) for v in v_list], key=lambda data: data[0])[-1]
                    if majority_count > len(v_list)/2:
                        v = majority_v
                    else:
                        v = max([item for item in v_list if item is not None], default=None)
                    if v is not None:
                        self.db.put(k, v)
                    else:
                        vanished_entries |= set([k])
            
            # after having synchronized all changed entries, update the root key and refresh the local state cache
            self.db.put(self.root_key, appropriate_backend_state_content_key)
            self._backups = None
            self._state = None
            self._state_content_key = None
            
            # and finally remove the entries that are not required anymore
            for k in vanished_entries:
                try:
                    self.db.delete(k)
                except:
                    pass
            
            if not rc_everywhere:
                # we need to recompute the local RCs now
                self.scrub()
                
            # if versions have been synchronized from backends to local database, the journal is incomplete
            self.journal.clear()
            
        # update all backends that are not already up-to-date
        for (backend_id, backend) in backends.items():
            backend_state = backend_states[backend_id]
            
            # backend only needs an update if it does not already have the latest version
            if backend_state['version'] < self.state['version']:
                # get changes from journal
                changes = self.journal.get_changes(backend_state['version'], self.state['version'])
                
                if changes is not None:
                    if not backend.store_rc:
                        # exclude reference counters
                        for op, ks in changes:
                            ks[:] = [k for k in ks if (len(k) % 2 == 0 or bytes((k[0], )) != b'r')]
                        changes[:] = [(op, ks) for (op, ks) in changes if ks]

                    # annotate put operations with actual contents and send changes to backend
                    changes = [(op, [(k, self.get_fn(k)) for k in ks] if op == 'put' else ks) for (op, ks) in changes]
                    
                    backend.post(changes)
    
                else: # changes is None
                    # if changes could not be determined from journal, a complete sync is required
    
                    # this should only happen if the user agrees
                    if not force: raise Exception('Out of sync: Incremental update of backend {} is not possible'.format(backend_id))

                    # to update the backend state, we first determine which entries are present at the backend
                    backend_keys = set(backend.list())
                    
                    # synchronization is then done in four steps:
                    # 1. we copy all immutable entries to the backend that do not already exist
                    # 2. if RCs shall be stored at the backend, we copy all mutable entries from the local database to the backend except the root entry
                    #    (in case of an error, this ensures that the backend is still in a consistent state)
                    # 3. we copy the root entry
                    # 4. we remove all backend entries that are not required anymore
                    local_keys = set(self.db.list())
                    changed_entries = [(k, self.get_fn(k)) for k in local_keys - backend_keys if len(k) % 2 == 0] # immutable entries
                    if backend.store_rc:
                        changed_entries.extend([(k, self.get_fn(k)) for k in local_keys if len(k) % 2 == 1 and k != self.root_key]) # mutable entries
                    changed_entries.append((self.root_key, self.get_fn(self.root_key))) # root entry
                    delete_entries = list(backend_keys - local_keys)
                    backend.post([('put', changed_entries), ('delete', delete_entries)])
        
        # if all backends are up-to-date now (which is the case if no specific backend_id was synced), we can clear the journal
        if backend_ids is None:
            self.journal.clear()
        
    '''
    Helpers.
    '''

    @classmethod
    def get_database_class(cls, database_type):
        for db_cls in AbstractDatabase.__subclasses__(): #@UndefinedVariable
            if db_cls.name == database_type:
                return db_cls

    @classmethod
    def get_db_wrapper(cls, put_fn, get_fn, delete_fn, list_fn):

        class DBWrapper():
            __slots__ = []
            
            def __setitem__(self, k, v):
                return put_fn(k, v)
            
            def __getitem__(self, k):
                return get_fn(k)
            
            def __delitem__(self, k):
                return delete_fn(k)
            
            def __contains__(self, k):
                return get_fn(k) != None
            
            def __iter__(self):
                for key in list_fn():
                    yield key
        
        return DBWrapper()

    @classmethod
    def get_rc_wrapper(cls, put_fn, get_fn, delete_fn, list_fn, journal=None, prefix=b'r'):

        if get_fn and put_fn and delete_fn:
            # define rc_inc / rc_dec interface to database
            rc_get = lambda key: struct.unpack('!Q', get_fn(key) or b'\x00\x00\x00\x00\x00\x00\x00\x00')[0]
            rc_set = lambda key, value: put_fn(key, struct.pack('!Q', value)) if value > 0 else delete_fn(key)
            def rc_inc_fn(key):
                new_count = rc_get(key) + 1
                rc_set(key, new_count)
                return new_count
            def rc_dec_fn(key):
                new_count = rc_get(key) - 1
                rc_set(key, new_count)
                return new_count
            
            # journalize RC functions
            if journal is not None:
                rc_inc_fn = journal.journalize_fn(rc_inc_fn, '+')
                rc_dec_fn = journal.journalize_fn(rc_dec_fn, '-')
                
            class RCWrapper():
                __slots__ = ['prefix']
                
                def __init__(self, prefix):
                    self.prefix = prefix
                
                def inc(self, k):
                    return rc_inc_fn(prefix + k)
                
                def dec(self, k):
                    return rc_dec_fn(prefix + k)
                
                def get(self, k):
                    return rc_get(prefix + k)
                
                def set(self, k, v):
                    rc_set(prefix + k, v)
                    return prefix + k
                
                def __iter__(self):
                    for k in list_fn():
                        if len(k) % 2 == 1 and bytes((k[0], )) == self.prefix:  # odd-length keys with prefix are reference counters by definition
                            yield k[len(self.prefix):]
        else:
            # use dummy RCWrapper
            class RCWrapper():
                __slots__ = ['prefix']
                
                def __init__(self, prefix):
                    self.prefix = prefix
                
                def inc(self, _):
                    return 1
                
                def dec(self, _):
                    return 1
                
                def get(self, _):
                    return 1
                
                def set(self, k, v):
                    raise Exception('not supported in grow_forever mode')
                
                def __iter__(self):
                    raise Exception('not supported in grow_forever mode')
        
        return RCWrapper(prefix)
    
    @classmethod
    def get_memory_rc_wrapper(cls, prefix):
        ''' Returns an RC wrapper which acts like a normal rc wrapper, but keeps its values only in memory.
            (Used by scrub to recalculate reference counters without changing the database.)
        '''
        class RCWrapper():
            
            __slots__ = ['prefix', '_data']
            
            def __init__(self, prefix):
                self.prefix = prefix
                self._data = defaultdict(int)
            
            def inc(self, k):
                self._data[k] += 1
                return self._data[k]
            
            def dec(self, k):
                self._data[k] -= 1
                new_value = self._data[k]
                if new_value == 0:
                    del self._data[k]
                return new_value
            
            def get(self, k):
                if k in self._data:
                    return self._data[k]
                else:
                    return 0
            
            def set(self, k, v):
                if v:
                    self._data[k] = v
                elif k in self._data:
                    del self._data[k]
            
            def __iter__(self):
                for key in self._data.keys():
                    yield key

        return RCWrapper(prefix)

    '''
    Implementation details.
    '''

    # TODO: move compression operations to the sec-cs library
    def _maybe_compress(self, value):
        if value is None:
            return value

        compress_obj = zlib.compressobj()
        compressed_value = []
        
        # compress individual chunks
        chunker = self.rolling_hash.create_chunker(chunk_size=self.chunk_size)
        pos = 0
        for chunk_boundary in chunker.next_chunk_boundaries(value):
            chunk = value[pos:chunk_boundary]
            compressed_value.append(compress_obj.compress(chunk))
            compressed_value.append(compress_obj.flush(zlib.Z_FULL_FLUSH))
            pos = chunk_boundary

        # compress remainder
        compressed_value.append(compress_obj.compress(value[pos:]))
        compressed_value.append(compress_obj.flush())
        
        return b''.join(compressed_value)
    
    # TODO: move compression operations to the sec-cs library
    def _maybe_decompress(self, value):
        if value is None:
            return value
        
        return zlib.decompress(value)

    def _get_total_size_of_path(self, path, total_size=0):
        # recursively determine size of a file system subtree
        if os.path.isdir(path):
            for name in os.listdir(path):
                subpath = os.path.join(path, name)
                total_size = self._get_total_size_of_path(subpath, total_size)
        total_size += os.path.getsize(path)
        return total_size

    def _add_progress(self, progress_bytes, progress_fn=None):
        # helper function to keep track of the current action's progress
        self._progress_current_bytes += progress_bytes
        if progress_fn: progress_fn(self._progress_current_bytes, self._progress_total_bytes)

    def _do_add_backend(self, backend_id, backend_type, backend_conf, store_rc=False):
        # create backend instance
        for cls in AbstractBackend.__subclasses__(): #@UndefinedVariable
            if cls.name == backend_type:
                self.backends[backend_id] = cls(self.chunk_size, self.seccs_f._crypto_wrapper, self.seccs_d._crypto_wrapper, self.seccs_b._crypto_wrapper, self.rolling_hash, backend_conf, store_rc=store_rc)
                break
        else:
            raise Exception('Unsupported backend type: {}'.format(backend_type))

    def _do_backup_path(self, path, base_key=None, progress_fn=None): # path is required to be an absolute path
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                content_key = self.seccs_f.put_content(self._maybe_compress(f.read()), ignore_rc=True)

            self._add_progress(os.path.getsize(path), progress_fn=progress_fn)
                
            # free some memory
            self.db.savepoint()
            if self.journal: self.journal.flush()

            return content_key
        
        elif os.path.isdir(path):
            # get existing entries if possible
            existing_entries = list_to_dict(msgpack.unpackb(self._maybe_decompress(self.seccs_d.get_content(base_key)), encoding='utf-8') if base_key is not None else [])
            
            # get current entries
            names = sorted(os.listdir(path))
            entries = {}
            for name in names:
                subpath = os.path.join(path, name)
                
                # determine existing entry
                existing_entry = existing_entries.get(name, None)
                
                try:
                    # determine meta data
                    st = os.lstat(subpath)
                    entry = dict((key, getattr(st, key)) for key in ('st_mode', 'st_uid', 'st_gid', 'st_size', 'st_mtime', 'st_ctime'))
                    entry['is_file'] = os.path.isfile(subpath)
                    
                    # if file is present in previous backup and has not changed, reuse
                    if existing_entry is not None and entry['is_file'] and set(entry.items()) - set(existing_entry.items()) == set():
                        entry['key'] = existing_entry['key']
                        self._add_progress(self._get_total_size_of_path(subpath), progress_fn=progress_fn)
                    else:
                        # backup recursively
                        entry['key'] = self._do_backup_path(subpath, existing_entry['key'] if existing_entry is not None else None, progress_fn=progress_fn)
                    
                    entries[name] = entry
                except IOError:
                    logger.error("{} not backed up due to IOError".format(subpath))
                
            # store directory representation
            directory_key, is_new = self.seccs_d.put_content_and_check_if_new(self._maybe_compress(msgpack.packb(dict_to_list(entries), use_bin_type=True)), ignore_rc=True)
            if is_new:
                for entry in entries.values():
                    seccs = self.seccs_f if entry['is_file'] else self.seccs_d
                    seccs._reference_counter.inc(entry['key'][:self.digest_size])
            self._add_progress(os.path.getsize(path), progress_fn=progress_fn)
            
            # free some memory
            self.db.savepoint()
            if self.journal: self.journal.flush()
            
            return directory_key
        
        else:
            raise NotImplementedError(path)

    def _do_recover_path(self, base_key, path, force=False, progress_fn=None): # path is required to be an absolute path
        # create directory if not existent
        if not os.path.exists(path): os.makedirs(path)
        
        # ensure that directory is empty, otherwise abort
        if len(os.listdir(path)) > 0 and not force:
            raise Exception('Recovery directory {path} must be empty'.format(path=path))
        
        # get existing entries
        existing_entries = list_to_dict(msgpack.unpackb(self._maybe_decompress(self.seccs_d.get_content(base_key)), encoding='utf-8'))
        
        # restore entries
        for name, entry in existing_entries.items():
            
            subpath = os.path.join(path, name)
            
            if entry['is_file']:
                # restore file content
                content = self._maybe_decompress(self.seccs_f.get_content(entry['key']))
                with open(subpath, 'wb') as f:
                    f.write(content)

                self._add_progress(os.path.getsize(subpath), progress_fn=progress_fn)
            
            else:
                # recursively restore subdirectory
                self._do_recover_path(entry['key'], subpath, force, progress_fn=progress_fn)

            # restore meta data
            if 'st_mode' in entry:
                os.chmod(subpath, entry['st_mode'])
            if 'st_uid' in entry and 'st_gid' in entry:
                try:
                    os.chown(subpath, entry['st_uid'], entry['st_gid']) #@UndefinedVariable
                except AttributeError:
                    # we don't care if this is not supported by the current platform
                    pass
            if 'st_mtime' in entry:
                os.utime(subpath, (entry['st_mtime'], entry['st_mtime']))

            self._add_progress(os.path.getsize(path), progress_fn=progress_fn)

    def _do_delete_backup(self, base_key):
        # get existing entries
        existing_entries = list_to_dict(msgpack.unpackb(self._maybe_decompress(self.seccs_d.get_content(base_key)), encoding='utf-8'))
        
        # delete entries
        for entry in existing_entries.values():
            key = entry['key']
            
            if entry['is_file']:
                self.seccs_f.delete_content(key)
            else:
                if self.seccs_d._reference_counter.dec(key[:self.digest_size]) == 0:
                    self._do_delete_backup(key)
        
        # delete base
        self.seccs_d.delete_content(base_key, ignore_rc=True)

    def _get_backups(self):
        if self._backups is None:
            self._backups = msgpack.unpackb(self._maybe_decompress(self.seccs_b.get_content(self.state['backups_key'])), encoding='utf-8') if self.state['backups_key'] is not None else []
        return self._backups
    
    backups = property(_get_backups)

    def _get_state(self):
        if self._state is None:
            self._state = {'backups_key': None, 'version': 0}
            if self.state_content_key is not None:
                try:
                    self._state = list_to_dict(msgpack.unpackb(self._maybe_decompress(self.seccs_b.get_content(self.state_content_key)), encoding='utf-8'))
                except:
                    pass
        return self._state
    
    state = property(_get_state)

    def _get_state_content_key(self):
        if self._state_content_key is None:
            self._state_content_key = self.get_fn(self.root_key)
        return self._state_content_key
    
    state_content_key = property(_get_state_content_key)

    def _persist_state(self):
        # first persist backups
        old_backups_key = self.state['backups_key']
        self.state['backups_key'] = self.seccs_b.put_content(self._maybe_compress(msgpack.packb(self.backups, use_bin_type=True)))
        
        # then persist state object that references the backup
        old_state_content_key = self.state_content_key
        self.state['version'] += 1
        self._state_content_key = self.seccs_b.put_content(self._maybe_compress(msgpack.packb(dict_to_list(self.state), use_bin_type=True)))
        self.put_fn(self.root_key, self._state_content_key)
        
        # finally delete old keys
        if old_backups_key is not None:
            self.seccs_b.delete_content(old_backups_key)
        if old_state_content_key is not None:
            self.seccs_b.delete_content(old_state_content_key)
            
        # journal commit
        if self.journal is not None:
            self.journal.commit(self.state['version'])
            
            # unless there are backends, the journal is not required
            if not self.backends:
                self.journal.clear()
                
        # compact database
        self.db.compact()

if __name__ == '__main__':
    '''
    Define helper functions.
    '''

    def initialize_triviback(config_path):
        if not os.path.exists(config_path):
            raise CommandError('Initialization required. Run init first.')
        print('Running triviback with configuration file {}...'.format(config_path))
        print()
        return Triviback(config_path)

    '''
    Define CLI actions.
    '''
    
    def define_basic_cli_actions():
        
        @arg('--journal-path', help='journal file', default=None)
        @arg('--db-path', help='database path', default=None)
        @arg('--db-type', choices=['semidbm', 'sqlite3', 'zodb'], default='zodb', help='database type')
        @arg('--grow-forever', default=False, help='disable reference counting mechanism for maximum security (completely prevents storage reclamation on backup deletion)')
        @arg('--chunk-size', help='target chunk size for chunking strategy', type=int)
        @arg('--window-size', help='rolling hash window size for chunking strategy', type=int)
        @arg('--use-compression', help='use zlib compression', default=False)
        def init(config_path=None, journal_path=None, db_path=None, db_type=None, grow_forever=None, chunk_size=None, window_size=None, use_compression=None):
            "Initialize the triviback database"
            if os.path.exists(config_path):
                raise CommandError('Triviback is already initialized. Remove configuration file if you want to re-run init.')
            
            Triviback(config_path=config_path, journal_path=journal_path, database_conf=db_path, database_type=db_type, chunk_size=chunk_size, window_size=window_size, grow_forever=grow_forever, use_compression=use_compression)
            yield 'Triviback database successfully initialized.'
    
        @arg('path', help='path to be backed up')
        @arg('--non-incremental', help='prevents triviback from speeding up the backup creation process by looking at changes with respect to a prior backup', default=False)
        def backup(path, config_path=None, non_incremental=None):
            "Create a new backup of a given path"
            triviback = initialize_triviback(config_path)
            
            def print_progress_fn(progress_current_bytes, progress_total_bytes):
                print('\rBackup in progress... {}({} / {} bytes)'.format('' if not progress_total_bytes else '{:.2f}% '.format(min(100, 100.0*progress_current_bytes/progress_total_bytes)), progress_current_bytes, progress_total_bytes), end='')

            try:
                backup_id = triviback.backup_path(path, not non_incremental, progress_fn=print_progress_fn)
            except Exception as e:
                raise CommandError(e)
            
            yield ''
            yield ''
            yield 'Successfully created a backup with ID {}'.format(backup_id)
    
        @arg('backup_id', help='ID of backup that is to be recovered', type=int)
        @arg('path', help='path the backup should be recovered to')
        @arg('--force', help='allow recover into non-empty directory', default=False)
        def recover(backup_id, path, config_path=None, force=None):
            "Recover a backup stored under an ID to a specific path"
            triviback = initialize_triviback(config_path)

            def print_progress_fn(progress_current_bytes, progress_total_bytes):
                print('\rRecovered {} bytes...'.format(progress_current_bytes), end='')
            
            try:
                triviback.recover_path_by_id(backup_id, path, force, progress_fn=print_progress_fn)
            except Exception as e:
                raise CommandError(e)
            
            yield ''
            yield ''
            yield 'Successfully recovered backup with ID {} to path {}'.format(backup_id, path)

        def list(config_path=None):
            "List all stored backups"
            triviback = initialize_triviback(config_path)
            backups = triviback.backups
            if backups:
                yield 'Existing backups:'
                backups = [(backup_id, datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'), path) for (backup_id, timestamp, path, _) in backups]
                yield tabulate(backups, ['ID', 'Creation time', 'Original path'], tablefmt='simple')
            else:
                yield 'No backups in database.'
    
        @arg('backup_id', help='ID of backup that is to be deleted', type=int)
        def delete(backup_id, config_path=None):
            "Delete backup by id"
            triviback = initialize_triviback(config_path)
            try:
                triviback.delete_backup_by_id(backup_id)
            except Exception as e:
                raise CommandError(e)
            yield 'Successfully deleted backup with ID {} '.format(backup_id)

        def scrub(config_path=None):
            "Remove stale objects from triviback's database and recalculate reference counters if enabled"
            triviback = initialize_triviback(config_path)
            triviback.scrub()
            yield 'Scrub finished.'
    
        return [init, backup, recover, list, delete, scrub]

    def define_backend_cli_actions():

        @arg('--backend-type', choices=['server', 'memory'], default='server', help='backend type')
        @arg('--backend-conf', default=None, help='backend configuration (e.g., URL of server backend)')
        @arg('--store-rc', default=False, help='store copies of reference counters at backends (allows slightly faster recovery in some situations at the expense of wasting storage space)')
        def add(config_path=None, backend_type=None, backend_conf=None, store_rc=None):
            "Add a new backend"
            triviback = initialize_triviback(config_path)
            try:
                triviback.add_backend(backend_type, backend_conf, store_rc=store_rc)
            except Exception as e:
                raise CommandError(e)
            yield 'Backend added successfully'
        
        def list(config_path=None):
            "List exsting backends"
            triviback = initialize_triviback(config_path)
            backends = triviback.backends
            if backends:
                yield 'Existing backends:'
                yield tabulate(backends.items(), ['ID', 'Backend'], tablefmt='simple')
            else:
                yield 'No backends configured.'
    
        @arg('--force', help='correct local or remote inconsistencies', default=False)
        def sync(config_path=None, backend_ids=None, force=None):
            "Synchronise local database <-> backends"
            triviback = initialize_triviback(config_path)
            try:
                triviback.sync_backends(backend_ids, force)
                yield 'Synchronization successful'
            except Exception as e:
                raise CommandError(e)
    
        @arg('backend_id', help='ID of backend that is to be deleted', type=int)
        def delete(backend_id, config_path=None):
            "Remove a backend"
            triviback = initialize_triviback(config_path)
            try:
                triviback.delete_backend(backend_id)
            except Exception as e:
                raise CommandError(e)
            yield 'Successfully deleted backend with ID {} '.format(backend_id)
    
        return [add, list, sync, delete]
    
    '''
    Set up CLI.
    '''

    parser = argparse.ArgumentParser(description='The triviback backup system.')

    global_decorators = [
            arg('--config-path', help='configuration file', default=os.path.expanduser(os.path.join('~', '.triviback', 'triviback.yaml')))
        ]

    # set up standard commands
    fns = define_basic_cli_actions()
    for decorator in global_decorators: fns = map(decorator, fns)
    argh.add_commands(parser, fns)

    # set up backend commands
    fns = define_backend_cli_actions()
    for decorator in global_decorators: fns = map(decorator, fns)
    argh.add_commands(parser, fns, namespace='backends', description="Manage backends")
    
    # finish CLI setup
    argh.dispatch(parser)
