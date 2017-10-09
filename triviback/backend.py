import base64

import seccs

try:
    import msgpack
except ImportError:
    msgpack = None

try:
    import requests
except ImportError:
    requests = None

import logging

logger = logging.getLogger(__name__)

class AbstractBackend(object):

    name = None
    
    def __init__(self, chunk_size, crypto_wrapper_f, crypto_wrapper_d, crypto_wrapper_b, rolling_hash, conf=None, store_rc=False):
        self.store_rc = store_rc
        
        # initialize storage data structure
        self.seccs_f = seccs.SecCSLite(chunk_size, self, crypto_wrapper_f, rolling_hash)
        self.seccs_d = seccs.SecCSLite(chunk_size, self, crypto_wrapper_d, rolling_hash)
        self.seccs_b = seccs.SecCSLite(chunk_size, self, crypto_wrapper_b, rolling_hash)

    def open(self):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()
    
    def put(self, k, v):
        raise NotImplementedError()
    
    def get(self, k):
        raise NotImplementedError()
    
    def delete(self, k):
        raise NotImplementedError()

    def post(self, commands):
        results = []
        for action, elements in commands:
            if action == 'put':
                for (k, v) in elements:
                    self.put(k, v)
            elif action == 'delete':
                for k in elements:
                    self.delete(k)
            elif action == 'get':
                results.extend([self.get(k) for k in elements])
            else:
                raise Exception('Action not implemented: {}'.format(action))
        return results

    def list(self):
        raise NotImplementedError()

    ''' implement dict interface '''

    def __getitem__(self, k):
        return self.get(k)

    def __setitem__(self, k, v):
        return self.put(k, v)

    def __delitem__(self, k):
        self.delete(k)
    
    def __contains__(self, k):
        return self.get(k) != None


class MemoryBackend(AbstractBackend):
    
    name = 'memory'
    
    def __init__(self, chunk_size, crypto_wrapper_f, crypto_wrapper_d, crypto_wrapper_b, rolling_hash, conf=None, store_rc=False):
        self._dict = dict()
        AbstractBackend.__init__(self, chunk_size, crypto_wrapper_f, crypto_wrapper_d, crypto_wrapper_b, rolling_hash, conf=conf, store_rc=store_rc)
        
    def __str__(self):
        return 'Oblivious memory backend (only for debugging)'
        
    def put(self, k, v):
        self._dict[k] = v
        
    def get(self, k):
        return self._dict.get(k, None)
    
    def delete(self, k):
        if k in self._dict:
            del self._dict[k]

    def open(self):
        pass
    
    def close(self):
        pass
    
    def list(self):
        return list(self._dict.keys())
    
    def dump(self):
        return self._dict

    def clear(self):
        self._dict.clear()

if not msgpack or not requests:
    logger.warning('ServerBackend not available due to missing libraries. Required: msgpack, requests')
else:
    class ServerBackend(AbstractBackend):
        
        name = 'server'
        
        def __init__(self, chunk_size, crypto_wrapper_f, crypto_wrapper_d, crypto_wrapper_b, rolling_hash, conf=None, store_rc=False):
            # configuration is required
            if not conf:
                raise Exception('Server backend requires configuration (specify via --backend-conf)')
            if isinstance(conf, str):
                conf = {'url': conf} # FIXME: support more complex configuration options via CLI
            self._url_prefix = conf['url'].rstrip('/') + '/'
            AbstractBackend.__init__(self, chunk_size, crypto_wrapper_f, crypto_wrapper_d, crypto_wrapper_b, rolling_hash, conf=conf, store_rc=store_rc)
    
        def __str__(self):
            return 'Server backend at ' + self._url_prefix
            
        def put(self, k, v):
            r = requests.put(self._url_prefix + base64.urlsafe_b64encode(k).decode('ascii'), data=v)
            if r.status_code != 200:
                raise Exception(repr(r))
            
        def get(self, k):
            r = requests.get(self._url_prefix + base64.urlsafe_b64encode(k).decode('ascii'))
            if r.status_code == 200:
                return r.content
            elif r.status_code == 404:
                return None
            else:
                raise Exception(repr(r))
        
        def delete(self, k):
            r = requests.delete(self._url_prefix + base64.urlsafe_b64encode(k).decode('ascii'))
            if r.status_code not in [200, 404]:
                raise Exception(repr(r))
    
        def open(self):
            pass
        
        def close(self):
            pass
    
        def post(self, commands):
            r = requests.post(self._url_prefix, data=msgpack.packb(commands, use_bin_type=True))
            if r.status_code == 200:
                return msgpack.unpackb(r.content)
            else:
                raise Exception(r)
        
        def list(self):
            r = requests.get(self._url_prefix)
            if r.status_code == 200:
                return msgpack.unpackb(r.content)
