import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
function walk(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name).replaceAll('\\','/')])}
const names=['package.json','package-lock.json','tsconfig.json','vite.config.ts','index.html',...walk('src')].sort();
const digest=crypto.createHash('sha256');
for(const name of names){digest.update(name);digest.update('\0');digest.update(fs.readFileSync(name));digest.update('\0')}
const files={};
for(const name of walk('dist').filter(p=>!p.endsWith('/build.json'))){files[name.slice(5)]=crypto.createHash('sha256').update(fs.readFileSync(name)).digest('hex')}
fs.writeFileSync('dist/build.json',JSON.stringify({source_sha256:digest.digest('hex'),files},null,2));
