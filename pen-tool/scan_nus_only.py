import asyncio
from bleak import BleakScanner

# Nordic UART Service UUID (Adafruit BLEUart)
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"

async def main():
    print("Scanning for BLE devices that advertise Nordic UART (NUS)...")

    # return_adv=True so we get AdvertisementData objects
    devices = await BleakScanner.discover(return_adv=True, timeout=8.0)

    if not devices:
        print("No BLE devices found at all.")
        return

    found_any = False

    # devices is a dict: {address: (BLEDevice, AdvertisementData)}
    for dev, adv in devices.values():
        service_uuids = adv.service_uuids or []
        # Compare lowercased to be safe
        has_nus = any(u.lower() == NUS_SERVICE_UUID for u in service_uuids)

        if not has_nus:
            continue  # skip anything without the NUS service

        found_any = True
        print("----------------------------------------------------")
        print("NUS device found!")
        print(f"Address : {dev.address}")
        print(f"Name    : {dev.name}")
        print(f"RSSI    : {adv.rssi}")
        print(f"Services: {service_uuids}")
        print(f"Manuf   : {adv.manufacturer_data}")

    if not found_any:
        print("No devices with Nordic UART (NUS) found.")

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
