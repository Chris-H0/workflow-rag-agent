from dotenv import load_dotenv

def load_env():
    load_dotenv(".env", override=True)

if __name__ == "__main__":
    load_env()
