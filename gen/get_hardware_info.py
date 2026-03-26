import torch


def get_hardware_info():
    """获取当前硬件信息，包括GPU和CPU的相关参数

    返回值:
        dict: 包含硬件信息的字典
    """
    hardware_info = {}

    # 获取GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpus = []
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            gpus.append({
                'id': i,
                'name': gpu_name,
                'capability': gpu_capability
            })
        hardware_info['gpus'] = gpus
    else:
        hardware_info['gpus'] = []

    # 获取CPU信息
    cpu_info = {
        'cores': torch.get_num_threads(),
        'architecture': torch.__config__.show().split('\n')[0]  # 获取CPU架构信息
    }
    hardware_info['cpu'] = cpu_info

    return hardware_info

if __name__ == "__main__":
    info = get_hardware_info()
    print(info)