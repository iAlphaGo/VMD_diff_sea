# from src.config import load_config
# from src.train import train, test
# import argparse


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
#     args = parser.parse_args()
#     config = load_config(args.config)
#     print('Config loaded')
#     mode = config.MODE
#     if mode == 1:
#         train(config)
#     else:
#         test(config)s
# if __name__ == "__main__":
#     main()


"""
主程序 - 启动训练或测试
"""

import argparse
from src.config import load_config
from src.trainer import VMDEhancedTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml', help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='运行模式')
    parser.add_argument('--num_modes', type=int, default=4, help='VMD模态数')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置VMD模态数
    if args.num_modes:
        config.NUM_VMD_MODES = args.num_modes
    
    # 创建训练器
    trainer = VMDEhancedTrainer(config)
    
    # 运行
    if args.mode == 'test' or config.MODE == 0:
        trainer.test()
    else:
        trainer.train()

if __name__ == "__main__":
    main()