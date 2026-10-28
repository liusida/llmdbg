"use strict";
const test=require('node:test'), assert=require('node:assert/strict');
const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm');
const extensionRoot=path.resolve(__dirname,'..');
const source=fs.readFileSync(path.join(extensionRoot,'extension.js'),'utf8');
const starters=path.resolve(extensionRoot,'../starters');
const catalog=JSON.parse(fs.readFileSync(path.join(starters,'starters.json'),'utf8'));

function harness(initial={},failStat=false) {
  const files=new Map(Object.entries(initial)), opened=[], messages=[];
  const uri={joinPath:(base,...parts)=>path.join(base,...parts),file:value=>value};
  const workspace={workspaceFolders:[{uri:'/workspace',name:'test'}],
    fs:{async stat(file){if(failStat)throw Object.assign(new Error('denied'),{code:'NoPermissions'});
        if(!files.has(file))throw Object.assign(new Error('missing'),{code:'FileNotFound'});return {};},
      async createDirectory(){},
      async copy(from,to,options){assert.equal(options.overwrite,false);assert.equal(files.has(to),false);files.set(to,fs.readFileSync(from));},
      async writeFile(file,data){files.set(file,data);}},
    async openTextDocument(file){opened.push(file);return file;}};
  const vscode={Uri:uri,workspace,window:{async showTextDocument(){},showInformationMessage:text=>messages.push(text)}};
  const context={vscode,fs,path,__dirname:extensionRoot,Buffer};
  vm.runInNewContext(source.slice(source.indexOf('async function initializeProject()'),source.indexOf('function expandWorkspaceFolder(')),context);
  return {run:()=>context.initializeProject(),files,opened,messages};
}

test('packaged starters exactly match the curated allowlist, including the image',()=>{
  require('node:child_process').execFileSync(process.execPath,[path.join(__dirname,'stage-adapter.js')]);
  assert.equal(catalog.files.filter(f=>f.endsWith('.llmdbg.py')).length,5);
  const bundled=path.join(extensionRoot,'bundled/starters');
  for(const file of ['starters.json',...catalog.files])
    assert.deepEqual(fs.readFileSync(path.join(bundled,file)),fs.readFileSync(path.join(starters,file)));
  assert.equal(fs.existsSync(path.join(bundled,'Qwen35-2B-Vision.llmdbg.py')),false);
  assert.equal(fs.existsSync(path.join(bundled,'.vscode/settings.json')),false);
  assert.ok(catalog.files.includes('.vscode/launch.json'));
});
test('Create Starter Files copies five demos, image, launch config and opens GPT2',async()=>{
  const h=harness();await h.run();
  for(const file of catalog.files)
    assert.deepEqual(h.files.get(path.join('/workspace',file)),fs.readFileSync(path.join(starters,file)));
  assert.deepEqual(h.files.get('/workspace/.vscode/launch.json'),fs.readFileSync(path.join(starters,'.vscode/launch.json')));
  assert.deepEqual(h.opened,['/workspace/GPT2-Text.llmdbg.py']);
  assert.equal(h.files.size,7);
});
test('repeated creation preserves edited demos, images, and launch configs',async()=>{
  const existing={'/workspace/Qwen3-Chat.llmdbg.py':Buffer.from('user code'),
    '/workspace/images/cat.jpg':Buffer.from('user image'),
    '/workspace/.vscode/launch.json':Buffer.from('user config')};
  const h=harness(existing);await h.run();await h.run();
  for(const [file,value] of Object.entries(existing))assert.deepEqual(h.files.get(file),value);
  assert.equal(h.files.size,7);
});
test('permission errors are propagated rather than overwriting a destination',async()=>{
  const h=harness({},true);await assert.rejects(h.run(),/denied/);assert.equal(h.files.size,0);
});
