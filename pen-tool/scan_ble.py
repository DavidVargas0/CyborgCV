import asyncio
from bleak import BleakScanner

# Nordic UART Service UUID (what Adafruit BLEUart uses)
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"


async def main():
    print("Scanning for BLE devices (about 8 seconds)...")
    devices = await BleakScanner.discover(timeout=8.0)

    if not devices:
        print("No BLE devices found.")
        return

    for d in devices:
        uuids = d.metadata.get("uuids", [])
        is_nus = NUS_SERVICE_UUID in (uuids or [])

        tag = "*** POSSIBLE PEN ***" if is_nus else "                  "

        print(f"{tag}  {d.address}  |  {d.name}  |  uuids={uuids}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
