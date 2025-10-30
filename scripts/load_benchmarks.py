#!/usr/bin/env python3
"""
Load benchmark data from benchmarks.json into PostgreSQL.

This script reads the synthetic benchmark data from data/benchmarks.json
and inserts it into the PostgreSQL exported_summaries table.
"""

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:compass@localhost:5432/compass"
    )

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print(f"Database URL: {db_url}")
        print("\nMake sure PostgreSQL is running:")
        print("  make postgres-start")
        sys.exit(1)


def load_benchmarks_json():
    """Load benchmarks from JSON file."""
    json_path = Path(__file__).parent.parent / "data" / "benchmarks.json"

    if not json_path.exists():
        print(f"❌ Error: {json_path} not found")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data.get("benchmarks", [])


def generate_config_id(benchmark):
    """Generate a deterministic config_id from benchmark configuration."""
    # Create a hash from the configuration
    config_str = f"{benchmark['model_hf_repo']}_{benchmark['hardware']}_{benchmark['hardware_count']}_{benchmark['prompt_tokens']}_{benchmark['output_tokens']}"
    return hashlib.md5(config_str.encode()).hexdigest()


def prepare_benchmark_for_insert(benchmark):
    """Prepare a benchmark record for database insertion."""
    from datetime import datetime

    prepared = benchmark.copy()

    # Generate UUID and config_id
    prepared['id'] = str(uuid.uuid4())
    prepared['config_id'] = generate_config_id(benchmark)

    # Add required fields with defaults
    prepared['type'] = 'local'  # benchmark type
    prepared['jbenchmark_created_at'] = datetime.now()
    prepared['created_at'] = datetime.now()
    prepared['updated_at'] = datetime.now()

    return prepared


def insert_benchmarks(conn, benchmarks):
    """Insert benchmarks into the database."""
    cursor = conn.cursor()

    # Clear existing synthetic data
    print("Clearing existing benchmark data...")
    cursor.execute("TRUNCATE TABLE exported_summaries RESTART IDENTITY CASCADE;")

    # Prepare benchmarks with required fields
    prepared_benchmarks = [prepare_benchmark_for_insert(b) for b in benchmarks]

    # Prepare insert query
    insert_query = """
        INSERT INTO exported_summaries (
            id,
            config_id,
            model_hf_repo,
            type,
            hardware,
            hardware_count,
            framework,
            framework_version,
            mean_input_tokens,
            mean_output_tokens,
            prompt_tokens,
            prompt_tokens_stdev,
            output_tokens,
            output_tokens_stdev,
            ttft_mean,
            ttft_p90,
            ttft_p95,
            ttft_p99,
            itl_mean,
            itl_p90,
            itl_p95,
            itl_p99,
            e2e_mean,
            e2e_p90,
            e2e_p95,
            e2e_p99,
            requests_per_second,
            tokens_per_second,
            jbenchmark_created_at,
            created_at,
            updated_at
        ) VALUES (
            %(id)s,
            %(config_id)s,
            %(model_hf_repo)s,
            %(type)s,
            %(hardware)s,
            %(hardware_count)s,
            %(framework)s,
            %(framework_version)s,
            %(mean_input_tokens)s,
            %(mean_output_tokens)s,
            %(prompt_tokens)s,
            %(prompt_tokens_stdev)s,
            %(output_tokens)s,
            %(output_tokens_stdev)s,
            %(ttft_mean)s,
            %(ttft_p90)s,
            %(ttft_p95)s,
            %(ttft_p99)s,
            %(itl_mean)s,
            %(itl_p90)s,
            %(itl_p95)s,
            %(itl_p99)s,
            %(e2e_mean)s,
            %(e2e_p90)s,
            %(e2e_p95)s,
            %(e2e_p99)s,
            %(requests_per_second)s,
            %(tokens_per_second)s,
            %(jbenchmark_created_at)s,
            %(created_at)s,
            %(updated_at)s
        );
    """

    print(f"Inserting {len(prepared_benchmarks)} benchmark records...")
    execute_batch(cursor, insert_query, prepared_benchmarks, page_size=100)

    conn.commit()
    print(f"✓ Successfully inserted {len(benchmarks)} benchmarks")

    # Show some statistics
    cursor.execute("""
        SELECT
            COUNT(DISTINCT model_hf_repo) as num_models,
            COUNT(DISTINCT hardware) as num_hardware_types,
            COUNT(DISTINCT (prompt_tokens, output_tokens)) as num_traffic_profiles,
            COUNT(*) as total_benchmarks
        FROM exported_summaries;
    """)
    stats = cursor.fetchone()

    print("\n📊 Database Statistics:")
    print(f"  Models: {stats[0]}")
    print(f"  Hardware types: {stats[1]}")
    print(f"  Traffic profiles: {stats[2]}")
    print(f"  Total benchmarks: {stats[3]}")

    # Show traffic profile distribution
    cursor.execute("""
        SELECT
            prompt_tokens,
            output_tokens,
            COUNT(*) as num_benchmarks
        FROM exported_summaries
        GROUP BY prompt_tokens, output_tokens
        ORDER BY prompt_tokens, output_tokens;
    """)

    print("\n🚦 Traffic Profile Distribution:")
    for row in cursor.fetchall():
        print(f"  ({row[0]}, {row[1]}): {row[2]} benchmarks")

    cursor.close()


def main():
    """Main function."""
    print("=" * 60)
    print("Loading Synthetic Benchmark Data into PostgreSQL")
    print("=" * 60)
    print()

    # Load benchmarks from JSON
    benchmarks = load_benchmarks_json()
    print(f"✓ Loaded {len(benchmarks)} benchmarks from JSON")

    # Connect to database
    print("Connecting to PostgreSQL...")
    conn = get_db_connection()
    print("✓ Connected to database")
    print()

    try:
        # Insert benchmarks
        insert_benchmarks(conn, benchmarks)
    except Exception as e:
        print(f"\n❌ Error inserting benchmarks: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  make postgres-query-traffic  # View traffic patterns")
    print("  make postgres-query-models   # View available models")
    print("  make postgres-shell          # Open PostgreSQL shell")


if __name__ == "__main__":
    main()
