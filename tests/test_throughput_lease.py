import fcntl
import importlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest

CHILD = '''
import importlib,json,os,sys
sys.path.insert(0,sys.argv[1]);H=importlib.import_module('experiments.batch_throughput')
path,original=sys.argv[2],int(sys.argv[3])
with H.lease(path,None if original<0 else original) as duplicate:
 print('ready',flush=True);sys.stdin.buffer.read(1)
 if original>=0: assert os.fstat(original).st_ino==os.fstat(duplicate.fileno()).st_ino
if original>=0: os.fstat(original)
'''

EOF = '''
import importlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);H=importlib.import_module('experiments.batch_throughput')
try:
 H.watch_lease_parent();print('ready',flush=True)
 while True:time.sleep(.01)
except KeyboardInterrupt:
 Path(sys.argv[2]).write_text(json.dumps(dict(status='cancelled',pid=os.getpid())))
'''


class ThroughputLeaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.H = importlib.import_module("experiments.batch_throughput")
        self.repo = str(Path(self.H.__file__).resolve().parents[1])
        self.lock = self.root / "lock"; self.lock.write_bytes(b"existing inode"); self.children = []
        self.addCleanup(self.reap)

    def reap(self):
        for child in self.children:
            if child.poll() is None: child.kill()
            child.communicate(timeout=5)

    def spawn(self, code, *arguments, pass_fds=()):
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", code, self.repo, *map(str, arguments)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=pass_fds, start_new_session=True)
        self.children.append(child)
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ); self.assertTrue(ready.select(5), "child did not become ready")
            self.assertEqual(child.stdout.readline(), b"ready\n")
        return child

    def test_shared_open_description_coexists_but_keeps_outsiders_excluded(self):
        before = self.lock.stat()
        with self.H.lease(self.lock) as original:
            descriptor = original.fileno()
            children = [self.spawn(CHILD, self.lock, descriptor, pass_fds=(descriptor,)) for _ in range(2)]
            outsider = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, self.repo, str(self.lock), "-1"],
                input=b"x", capture_output=True, timeout=5)
            self.assertNotEqual(outsider.returncode, 0); self.assertIn(b"BlockingIOError", outsider.stderr)
            for child in children[:1]:
                _, error = child.communicate(b"x", timeout=5); self.assertEqual(child.returncode, 0, error)
            self.assertEqual(os.fstat(descriptor).st_ino, before.st_ino)
            with self.lock.open("rb") as outsider:
                with self.assertRaises(BlockingIOError): fcntl.flock(outsider, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.lock.open("rb") as outsider:
            with self.assertRaises(BlockingIOError): fcntl.flock(outsider, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, error = children[1].communicate(b"x", timeout=5); self.assertEqual(children[1].returncode, 0, error)
        with self.H.lease(self.lock) as later: self.assertEqual(os.fstat(later.fileno()).st_ino, before.st_ino)
        self.assertEqual(self.lock.read_bytes(), b"existing inode")

    def test_inherited_descriptor_rejects_wrong_types_identity_and_symlinks(self):
        other = self.root / "other"; other.write_bytes(b"other inode")
        link = self.root / "alias"; link.symlink_to(self.lock)
        with self.lock.open("rb") as original, other.open("rb") as wrong:
            closed = os.dup(original.fileno()); os.close(closed)
            for value in (True, -1, 1.0, closed, wrong.fileno()):
                with self.subTest(value=value), self.assertRaises((ValueError, OSError, TypeError)):
                    with self.H.lease(self.lock, value): pass
            with self.assertRaises((ValueError, OSError)):
                with self.H.lease(link, original.fileno()): pass
            os.fstat(original.fileno()); os.fstat(wrong.fileno())
        with self.assertRaises((ValueError, OSError)):
            with self.H.lease(link): pass
        missing = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            with self.H.lease(missing): pass
        self.assertFalse(missing.exists())

    def test_parent_eof_interrupts_only_its_owned_cli_and_persists_receipt(self):
        path = self.root / "receipt.json"; child = self.spawn(EOF, path)
        child.stdin.close(); child.stdin = None
        _, error = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 0, error)
        self.assertEqual(json.loads(path.read_text()), dict(status="cancelled", pid=child.pid))
