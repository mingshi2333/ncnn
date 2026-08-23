// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "storezip.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

static int read_entry(const char* path, const char* name)
{
    pnnx::StoreZipReader reader;
    if (reader.open(path) != 0)
        return 1;

    if (!reader.has_file(name))
        return 1;

    const uint64_t size = reader.get_file_size(name);
    std::vector<char> data(size);
    if (reader.read_file(name, data.data()) != 0)
        return 1;

    if (!data.empty())
        fwrite(data.data(), 1, data.size(), stdout);

    return 0;
}

static int expect_open_failure(const char* path)
{
    pnnx::StoreZipReader reader;
    return reader.open(path) == 0 ? 1 : 0;
}

static int expect_read_failure_after_open(const char* path, const char* name)
{
    pnnx::StoreZipReader reader;
    if (reader.open(path) != 0)
        return 1;

    if (!reader.has_file(name))
        return 1;

    const uint64_t size = reader.get_file_size(name);
    std::vector<char> data(size);
    return reader.read_file(name, data.data()) == 0 ? 1 : 0;
}

static int expect_compressed_entry(const char* path, const char* name)
{
    pnnx::StoreZipReader reader;
    if (reader.open(path) != 0)
        return 1;

    if (!reader.has_file(name))
        return 1;

    return reader.is_file_stored(name) ? 1 : 0;
}

static int reopen_without_stale_entry(const char* first_path, const char* second_path, const char* stale_name)
{
    pnnx::StoreZipReader reader;
    if (reader.open(first_path) != 0)
        return 1;
    if (reader.open(second_path) != 0)
        return 1;

    const std::vector<std::string> names = reader.get_names();
    if (std::find(names.begin(), names.end(), stale_name) != names.end())
        return 1;

    return reader.has_file(stale_name) ? 1 : 0;
}

static int writer_round_trip(const char* path)
{
    static const char payload[] = "storezip-round-trip";

    pnnx::StoreZipWriter writer;
    if (writer.open(path) != 0)
        return 1;
    if (writer.write_file("payload", payload, sizeof(payload) - 1) != 0)
        return 1;
    if (writer.close() != 0)
        return 1;

    pnnx::StoreZipReader reader;
    if (reader.open(path) != 0)
        return 1;
    if (!reader.has_file("payload"))
        return 1;
    if (!reader.is_file_stored("payload"))
        return 1;
    if (reader.get_file_size("payload") != sizeof(payload) - 1)
        return 1;

    std::vector<char> actual(sizeof(payload) - 1);
    if (reader.read_file("payload", actual.data()) != 0)
        return 1;

    return memcmp(actual.data(), payload, actual.size()) == 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 3)
        return 2;

    const std::string command = argv[1];

    if (command == "read" && argc == 4)
        return read_entry(argv[2], argv[3]);
    if (command == "expect-open-failure" && argc == 3)
        return expect_open_failure(argv[2]);
    if (command == "expect-read-failure-after-open" && argc == 4)
        return expect_read_failure_after_open(argv[2], argv[3]);
    if (command == "expect-compressed-entry" && argc == 4)
        return expect_compressed_entry(argv[2], argv[3]);
    if (command == "reopen-without-stale-entry" && argc == 5)
        return reopen_without_stale_entry(argv[2], argv[3], argv[4]);
    if (command == "writer-round-trip" && argc == 3)
        return writer_round_trip(argv[2]);

    return 2;
}
