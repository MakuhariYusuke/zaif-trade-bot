import fs from 'fs';
import path from 'path';

export function writeFileAtomic(filePath: string, data: string | Buffer): void {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(
    dir,
    `.${path.basename(filePath)}.tmp-${process.pid}-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2)}`
  );
  const fd = fs.openSync(tmp, 'w');
  try {
    if (typeof data === 'string') {
      fs.writeFileSync(fd, data, 'utf8');
    } else {
      fs.writeFileSync(fd, data);
    }
    fs.fsyncSync(fd);
  } finally {
    try { fs.closeSync(fd); } catch { /* ignore */ }
  }
  fs.renameSync(tmp, filePath);
}
