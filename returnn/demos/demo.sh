#!/bin/bash

set -e
demo=$1
test -n "$demo" || { echo "usage: $0 <demo>"; exit 1; }
[[ "$demo" == demo-*.config ]] && demo="${demo:5}" && demo="${demo/.config/}"
shift

# Delete any old data.
rm /tmp/returnn.demo-${demo}.* 2>/dev/null || true
rm -rf /tmp/returnn 2>/dev/null || true
rm -rf /tmp/$(whoami)/returnn 2>/dev/null || true

cd $(dirname $0)
test -e demo-${demo}.config || { echo "error: demo-${demo}.config not found"; exit 1; }
echo "run: ../rnn.py demo-${demo}.config $*"
../rnn.py demo-${demo}.config $*

echo "finished. deleting models."
rm /tmp/returnn.demo-${demo}.network.* 2>/dev/null || true
rm -rf /tmp/returnn 2>/dev/null || true
