import json

class MyDictionaryManager:
    _instance = None

    def __new__(cls, source, file_name,process_id, problem_type):
        if cls._instance is None:
#             cls.folder_path = folder_path
            cls._instance = super(MyDictionaryManager, cls).__new__(cls)
            cls._instance.file_name = file_name
            cls._instance.process_id=  process_id
            cls._instance.problem_type = problem_type
            cls._instance.json_filename = f'Knowledge/{problem_type}/{source}/{process_id}/json/file_paths.json'
#             cls._instance.json_filename = f'{problem_type}/json/{file_name}.json'
            cls._instance.my_dict = {}
        return cls._instance

    def update_value(self, key, value):
        self.my_dict[key] = value

    def save_dictionary(self):
        with open(self.json_filename, 'w') as json_file:
            json.dump(self.my_dict, json_file, indent=4)

    def load_dictionary(self):
        try:
            with open(self.json_filename, 'r') as json_file:
                self.my_dict = json.load(json_file)
        except FileNotFoundError:
            pass
            
    @classmethod
    def close_instance(cls):
        cls._instance = None
