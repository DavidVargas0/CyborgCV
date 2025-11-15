import asyncio
from bleak import BleakScanner

# Nordic UART Service UUID (Adafruit BLEUart)
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"

async def main():
    print("Scanning for BLE devices (about 8 seconds)...")

    # return_adv=True gives us AdvertisementData objects as well
    devices = await BleakScanner.discover(return_adv=True, timeout=8.0)

    if not devices:
        print("No BLE devices found.")
        return

    found_any = False

    # devices is a dict: {address: (BLEDevice, AdvertisementData)}
    for dev, adv in devices.values():
        found_any = True

        service_uuids = adv.service_uuids or []
        is_nus = any(u.lower() == NUS_SERVICE_UUID for u in service_uuids)

        tag = "<<< POSSIBLE PEN (NUS) >>>" if is_nus else ""

        print("----------------------------------------------------")
        print(f"{tag}")
        print(f"Address : {dev.address}")
        print(f"Name    : {dev.name}")
        print(f"RSSI    : {adv.rssi}")
        print(f"Services: {service_uuids}")
        print(f"Manuf   : {adv.manufacturer_data}")

    if not found_any:
        print("No BLE devices found at all.")

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
