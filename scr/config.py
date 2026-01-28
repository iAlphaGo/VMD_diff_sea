# import yaml
# import os


# class Config(dict):
#     def __init__(self, config_path):
#         with open(config_path, 'r') as f:
#             self._yaml = f.read()
#             self._dict = yaml.safe_load(self._yaml)
#             self._dict['PATH'] = os.path.dirname(config_path)

#     def __getattr__(self, name):
#         if self._dict.get(name) is not None:
#             return self._dict[name]
#         return None

#     def print(self):
#         print('Model configurations:')
#         print('---------------------------------')
#         print(self._yaml)
#         print('')
#         print('---------------------------------')
#         print('')


# def load_config(path):
#     config_path = path
#     config = Config(config_path)
#     return config

# import yaml
# import os


# class Config(dict):
#     def __init__(self, config_path):
#         with open(config_path, 'r') as f:
#             self._yaml = f.read()
#             self._dict = yaml.safe_load(self._yaml)
#             self._dict['PATH'] = os.path.dirname(config_path)

#     def __getattr__(self, name):
#         if self._dict.get(name) is not None:
#             return self._dict[name]
#         return None

#     def print(self):
#         print('Model configurations:')
#         print('---------------------------------')
#         print(self._yaml)
#         print('')
#         print('---------------------------------')
#         print('')


# def load_config(path):
#     config_path = path
#     config = Config(config_path)
#     return config



"""
配置文件 - 添加VMD相关参数
"""

import yaml
import os

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
            
        # VMD相关参数（如果没有则设置默认值）
        if 'USE_VMD' not in self._dict:
            self._dict['USE_VMD'] = True
        if 'VMD_MODE' not in self._dict:
            self._dict['VMD_MODE'] = 'condition'  # 'preprocess', 'condition', 'multibranch'
        if 'NUM_VMD_MODES' not in self._dict:
            self._dict['NUM_VMD_MODES'] = 4
    
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        return None
    
    def print(self):
        print('模型配置:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('VMD参数:')
        print(f'  USE_VMD: {self.USE_VMD}')
        print(f'  VMD_MODE: {self.VMD_MODE}')
        print(f'  NUM_VMD_MODES: {self.NUM_VMD_MODES}')
        print('---------------------------------')
        print('')

def load_config(path):
    config_path = path
    config = Config(config_path)
    return config