import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for BLE devices for 8 seconds...")
    devices = await BleakScanner.discover(timeout=8.0)

    if not devices:
        print("No BLE devices found.")
        return

    print("\nFound these devices:\n")
    for d in devices:
        # Some devices don't have a name -> show 'None' explicitly
        print(f"{d.address:>20}  |  {d.name}")

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
