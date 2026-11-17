// Fast drafts only. Final paper/PPT previews come from native slide exports.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const runtime=process.env.ARTIFACT_NODE_MODULES || 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const require=createRequire(import.meta.url),sharp=require(path.join(runtime,'sharp'));
await fs.mkdir(path.join(root,'.draft'),{recursive:true});
for(const stem of process.argv.slice(2)){
 await sharp(path.join(root,stem+'.svg')).resize({width:1800}).png().toFile(path.join(root,'.draft',stem+'.png'));
 console.log(stem);
}
