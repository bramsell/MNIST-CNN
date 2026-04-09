"""Entry point — run this file directly to start the Flask server."""
from server.app import app
import yaml

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    s = cfg["server"]
    app.run(host=s["host"], port=s["port"])
