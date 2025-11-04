# rpi_client.py
# Captures an image from a Pi camera or reads from filesystem, and sends to server
# via HTTP POST. Also example of MQTT-based telemetry (publish predictions).

import requests
import time
import argparse
import os
from PIL import Image
import io
import paho.mqtt.client as mqtt

def send_image_http(img_path, server_url):
    files = {'file': open(img_path, 'rb')}
    r = requests.post(server_url.rstrip('/') + '/predict', files=files, timeout=10)
    return r.json()

def mqtt_publish(broker, topic, payload):
    client = mqtt.Client()
    client.connect(broker, 1883, 60)
    client.publish(topic, payload)
    client.disconnect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--server', default='http://<SERVER_IP>:5000')
    parser.add_argument('--mqtt_broker', default=None, help='Optional MQTT broker address')
    parser.add_argument('--mqtt_topic', default='plants/predictions')
    args = parser.parse_args()

    print("Sending", args.image, "to", args.server)
    try:
        resp = send_image_http(args.image, args.server)
        print("Server response:", resp)
        if args.mqtt_broker:
            import json
            mqtt_publish(args.mqtt_broker, args.mqtt_topic, json.dumps(resp))
            print("Published to MQTT:", args.mqtt_broker, args.mqtt_topic)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
