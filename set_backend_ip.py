import json
import sys

if __name__ == "__main__":
    new_backend_ip = sys.argv[1]
    with open("./alectio_sdk/flask_wrapper/config.json", "r") as file:
        json_data = json.load(file)
        if json_data["backend_ip"]:
            json_data["backend_ip"] = new_backend_ip
    with open("./alectio_sdk/flask_wrapper/config.json", "w") as file:
        json.dump(json_data, file, indent=2)
