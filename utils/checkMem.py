def check_mem():
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    # Convert to GB for readability
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    print(f"Free GPU memory: {free_gb:.2f} GB")
    print(f"Total GPU memory: {total_gb:.2f} GB")