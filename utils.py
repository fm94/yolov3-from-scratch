

def parse_config(config_file):
    " Read the official yolo config file correctly"
    with open(config_file) as f:
        lines = f.readlines()
    config = []
    for line in lines:
        if line and line != '\n' and line[0] != '#':
            stripped_line = line.replace(' ', '').replace('\n', '')
            if stripped_line[0] == '[':
                try:
                    config.append(data)
                except UnboundLocalError:
                    pass
                data = {}
                data['type'] = stripped_line[1:-1]
            else:
                [key, value] = stripped_line.split('=')
                data[key] = value
    config.append(data)
    network_info = config[0]
    config.pop(0)
    return config, network_info