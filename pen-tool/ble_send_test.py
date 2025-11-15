import asyncio
from bleak import BleakClient

# RX characteristic of Nordic UART Service
NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

# Your Feather's address:
DEVICE_ADDRESS = "EE:24:62:F7:4B:91"

async def main():
    print(f"Connecting to {DEVICE_ADDRESS}...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        print("Connected!")

        for code in ["0", "2", "8"]:
            print(f"Sending code {code}...")
            await client.write_gatt_char(NUS_RX_UUID, code.encode("ascii"))
            await asyncio.sleep(1.0)

        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
