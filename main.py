from utils.config import opt

def train(**kwargs):
    opt._parse(kwargs)

if __name__ == '__main__':
    import fire
    fire.Fire()