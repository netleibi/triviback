import argparse
import base64
import codecs
import os.path

from flask import Flask, request, abort
from flask.views import MethodView
import msgpack
import yaml

from triviback.database import AbstractDatabase


class TrivibackServer():
    
    def __init__(self, config_path=None, database_conf=None, database_type=None):
        
        # load or initialize configuration
        self.config_path = config_path
        self.triviback_config = yaml.safe_load(codecs.open(config_path, 'r', encoding='utf8')) if config_path is not None and os.path.isfile(config_path) else None
        if self.triviback_config is None: self.triviback_config = {}
        config_changed = False

        # database configuration
        try:
            try:
                database = self.triviback_config['database']
            except KeyError:
                database = {}
                self.triviback_config['database'] = database
                raise
            database_conf = database['conf']
            database_type = database['type']
        except KeyError:
            if database_type is None: database_type = 'memory'
            database['conf'] = database_conf
            database['type'] = database_type
            config_changed = True

        # load / initialize database
        db = None
        for cls in AbstractDatabase.__subclasses__(): #@UndefinedVariable
            if cls.name == database_type: db = cls(database_conf)
        if db is None:
            raise Exception('Unsupported database type: {}'.format(database_type))
        self.db = db
        
        # extract required functions from database
        put_fn = db.put
        get_fn = db.get
        delete_fn = db.delete

        # remember functions for internal use
        self.put_fn, self.get_fn, self.delete_fn = put_fn, get_fn, delete_fn

        # save config
        if config_changed: self.save_config()

    def __del__(self):
        # close database
        if self.db is not None:
            self.db.close()

    def save_config(self):
        # store configuration
        if self.config_path is not None:
            yaml.dump(self.triviback_config, codecs.open(self.config_path, 'w', encoding='utf8'), default_flow_style=False)

    '''
    Public interface.
    '''

    class TrivibackAPI(MethodView):

        def __init__(self, server):
            self.get_fn = server.get_fn
            self.put_fn = server.put_fn
            self.delete_fn = server.delete_fn
            self.list_fn = server.db.list
            self.compact_fn = server.db.compact

        # implement GET
        def get(self, k=None):
            if k is not None:
                k = base64.urlsafe_b64decode(k)
                v = self.get_fn(k)
                if v is None:
                    abort(404)
                else:
                    return v
            else: # k is None
                return msgpack.packb(self.list_fn(), use_bin_type=True)
            
        # implement PUT
        def put(self, k):
            k = base64.urlsafe_b64decode(k)
            v = request.data
            self.put_fn(k, v)
            return ''
    
        # implement DELETE
        def delete(self, k):
            k = base64.urlsafe_b64decode(k)
            try:
                self.delete_fn(k)
            except:
                abort(404)
            return ''
        
        # implement POST
        def post(self):
            # data must be msgpack-encoded data structure in format: [('put', [(k, v), (k, v), ...]), ('delete', [k, k, ...])]
            encoded_commands = request.data
            commands = msgpack.unpackb(encoded_commands)
            results = []
            for action, elements in commands:
                if isinstance(action, bytes):
                    action = action.decode('ascii')
                if action == 'put':
                    for (k, v) in elements:
                        self.put_fn(k, v)
                elif action == 'delete':
                    for k in elements:
                        try:
                            self.delete_fn(k)
                        except:
                            pass
                elif action == 'get':
                    for k in elements:
                        results.append(self.get_fn(k))
                else:
                    raise Exception('Unsupported action: {}'.format(action))
            self.compact_fn()
            return msgpack.packb(results, use_bin_type=True)
    
    # register provided API at a Flask App
    def register_on_app(self, app, endpoint='db_api'):
        view_func = TrivibackServer.TrivibackAPI(self).as_view(endpoint, self)
        app.add_url_rule('/<string:k>', view_func=view_func, methods=['GET', 'PUT', 'DELETE'])
        app.add_url_rule('/', view_func=view_func, methods=['GET', 'POST'])

def run_server(config_path=None, database_path=None, database_type='memory', host='127.0.0.1', port=1337, debug=False):
    triviback_server = TrivibackServer(config_path, database_path, database_type)

    ''' Create and run webserver for RESTful API '''
    app = Flask(__name__)
    triviback_server.register_on_app(app)
    app.run(debug=debug, host=host, port=port)

def main():
    parser = argparse.ArgumentParser(description='Runs a simple backend server providing a REST API for the __init__ backup system.')
    
    parser.add_argument('configpath', metavar='CONFIG_PATH', type=str, help='path of configuration file')
    parser.add_argument('--db-path', metavar='PATH', type=str, help='specifies the path to the database (only considered during initialization)', default=None)
    parser.add_argument('--db-type', metavar='TYPE', help='specifies the database type (only considered during initialization)', default='zodb', choices=['semidbm', 'sqlite3', 'zodb'])
    parser.add_argument('--debug', type=bool, help='enables debug output', default=False)
    parser.add_argument('--port', type=int, help='specifies the TCP port the server listens to', default=1337)
    
    args = parser.parse_args()
    
    run_server(config_path=args.configpath, database_path=args.db_path, database_type=args.db_type, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
