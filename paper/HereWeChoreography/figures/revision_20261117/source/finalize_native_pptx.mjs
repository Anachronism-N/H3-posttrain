import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const skill=process.env.PRESENTATIONS_SKILL_DIR || 'C:/Users/32665/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11011/skills/presentations';
const runtimeRoot='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies';
process.env.RUNTIME_NODE_MODULES ||= path.join(runtimeRoot,'node/node_modules');
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=path.join(root,'powerpoint/HereWeChoreography_figures_revision_v2.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const result=await finalizePresentation({
  explicitTotalSlideCount:8,requestedContentSlideCount:8,
  workspaceDir:root,candidatePath:path.join(root,'.pptx-build/candidate.pptx'),finalPath,
  pythonExecutable:path.join(runtimeRoot,'python/python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit'],
  requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'design',families:['Times New Roman']},
  verifyArtifactToolImport:true,
  receiptPath:path.join(root,'.pptx-build/final.validation.json')
});
console.log(JSON.stringify(result));
