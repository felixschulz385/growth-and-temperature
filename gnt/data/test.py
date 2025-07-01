import sqlite3
import asyncio


# read batch of 1000 rows from files table where status is 'pending'
def fetch_pending_batches(batch_size=1000):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE status = 'pending' LIMIT ?", (batch_size,))
    rows = cursor.fetchall()
    return rows

# a function to process each row
async def process_row(row):
    # Simulate processing time
    await asyncio.sleep(0.1)
    print(f"Processing row: {row[0]}")  # Assuming the first column is an identifier
    
# process rows in batches asynchronously
async def fetch_pending_batches_async(batch_size=1000):
    while True:
        rows = fetch_pending_batches(batch_size)
        if not rows:
            break  # No more pending rows
        tasks = [process_row(row) for row in rows]
        await asyncio.gather(*tasks)
        yield rows  # Yield the batch of rows for further processing
        
# Close the database connection when done
async def close_connection():
    conn.close()
        
# Example usage
async def main():
    async for batch in fetch_pending_batches_async():
        print(f"Processed batch of {len(batch)} rows")
    await close_connection()
    
# Run the main function
if __name__ == "__main__":
    # open database
    conn = sqlite3.connect('C:\\Users\\schulz0022\\hpc_data_index\\download_glass_modis_daily - Copy.sqlite')
    asyncio.run(main()) 