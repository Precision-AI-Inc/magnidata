#!/bin/sh
# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint for the api and trainer containers: run the command as the host user who owns the
# checkout, not as root, so what the container writes under data_user/ can be cleaned up without
# sudo. The user is MAGNIDATA_UID/MAGNIDATA_GID when set, otherwise the owner of the bind-mounted
# data/ folder. A root-owned checkout, or a container already started as another user, runs as is.
set -eu

owner_ref=/app/data
if [ "$(id -u)" != 0 ] || [ ! -e "$owner_ref" ]; then
    exec "$@"
fi
uid="${MAGNIDATA_UID:-$(stat -c %u "$owner_ref")}"
gid="${MAGNIDATA_GID:-$(stat -c %g "$owner_ref")}"
if [ "$uid" = 0 ]; then
    exec "$@"
fi

# Earlier versions ran as root; hand the files they left in the writable folders to the user.
for dir in /app/data_user /app/db; do
    if [ -d "$dir" ]; then
        find "$dir" -user 0 -exec chown -h "$uid:$gid" {} +
    fi
done

# The user has no passwd entry, so give it a writable home for caches (torch weights, matplotlib).
HOME=/tmp/magnidata-home
export HOME
mkdir -p "$HOME"
chown "$uid:$gid" "$HOME"

exec setpriv --reuid="$uid" --regid="$gid" --clear-groups "$@"
