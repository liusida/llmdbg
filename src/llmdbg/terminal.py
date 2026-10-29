"""Interactive and JSONL/scriptable terminal frontend for the existing DAP backend."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import sys
import threading
import time


class DAPError(RuntimeError):
    pass


class Client:
    def __init__(self, address, timeout=1800, output=None):
        self.socket=socket.create_connection(address,timeout=min(timeout,30))
        self.timeout=timeout;self.output=output or (lambda text:None)
        self.buffer=bytearray();self.sequence=0;self.events=[]

    def close(self):
        self.socket.close()

    def read(self,deadline):
        def receive():
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('DAP request timed out')
            self.socket.settimeout(remaining)
            chunk=self.socket.recv(65536)
            if not chunk:raise EOFError('DAP connection closed')
            self.buffer.extend(chunk)
        while b'\r\n\r\n' not in self.buffer:receive()
        header,_,_=self.buffer.partition(b'\r\n\r\n')
        length=int(next(line.split(b':',1)[1] for line in header.split(b'\r\n') if line.lower().startswith(b'content-length:')))
        if length<0 or length>128*1024*1024:raise DAPError('Invalid DAP message size')
        start=len(header)+4
        while len(self.buffer)<start+length:receive()
        message=json.loads(self.buffer[start:start+length]);del self.buffer[:start+length]
        if message.get('event')=='output':self.output(message.get('body',{}).get('output',''))
        return message

    def request(self,command,**arguments):
        self.sequence+=1
        payload=json.dumps(dict(seq=self.sequence,type='request',command=command,arguments=arguments)).encode()
        self.socket.sendall(f'Content-Length: {len(payload)}\r\n\r\n'.encode()+payload)
        deadline=time.monotonic()+self.timeout
        while True:
            message=self.read(deadline)
            if message.get('type')=='response':
                if message.get('request_seq')!=self.sequence:raise DAPError('Unexpected DAP response')
                if not message.get('success'):raise DAPError(message.get('message',f'{command} failed'))
                return message.get('body',{})
            if message.get('event') in ('stopped','terminated','exited','llmdbg/computationFailed'):
                self.events.append(message)

    def wait_stop(self):
        deadline=time.monotonic()+self.timeout
        while True:
            event=self.events.pop(0) if self.events else self.read(deadline)
            name=event.get('event')
            if name=='stopped':return dict(event='stopped',**event.get('body',{}))
            if name in ('terminated','exited'):return dict(event=name,**event.get('body',{}))
            if name=='llmdbg/computationFailed':raise DAPError(event.get('body',{}).get('message','Computation failed'))


HELP='''launch FILE                 Launch a .llmdbg.py file, paused at entry
break LINE|mlp|attention [token=N,layer=N]   Add a conditional breakpoint
clear                       Clear breakpoints in the current source
continue / c                Run to next breakpoint or termination
next / n                    Step over
step / s                    Step in
out                         Step out
where                       Show stack and source location
vars [REFERENCE]            Inspect scopes/variables
print / p EXPRESSION        Evaluate a debugger expression
tokens                      Show prompt and generated tokens
vector                      Inspect the active vector
lens EXPRESSION             Decode a tensor through all configured lenses (abc)
lens [j|r|logit] [VIEW]      Site-based readout (default view: residual-post)
request COMMAND JSON        Any DAP/custom request (does not wait for a stop)
quit / q                    Disconnect; external hosts keep their model resident
help                        Show this help'''


class Terminal:
    def __init__(self,client,args):
        self.client=client;self.args=args;self.source=None;self.breakpoints=[];self.calls={};self.launched=False

    def launch(self,filename=None):
        arguments={'stopOnEntry':True}
        if filename:
            from llmdbg.config import load_debug_config
            self.source=str(Path(filename).expanduser().resolve())
            config=load_debug_config(self.source);self.calls=config.get('callLines',{})
            arguments['config']=self.source
        elif not self.args.demo:raise ValueError('Use launch FILE or --demo')
        for name,key in [('model','model'),('device','device'),('dtype','dtype'),('max_new_tokens','maxNewTokens')]:
            value=getattr(self.args,name,None)
            if value is not None:arguments[key]=value
        self.client.events.clear();self.breakpoints=[]
        self.client.request('launch',**arguments);self.launched=True
        self.client.request('configurationDone')
        return self.stopped()

    def stopped(self):
        result=self.client.wait_stop()
        if result['event'] in ('terminated','exited'):
            self.launched=False
        if result['event']=='stopped':
            frames=self.client.request('stackTrace',threadId=1).get('stackFrames',[])
            if frames:
                result['frame']=frames[0]
                self.source=frames[0].get('source',{}).get('path',self.source)
        return result

    def execute(self,line):
        command,_,rest=line.strip().partition(' ');rest=rest.strip()
        command={'c':'continue','n':'next','s':'step','p':'print','b':'break','q':'quit'}.get(command,command)
        if command=='help':return {'help':HELP}
        if command=='quit':return {'quit':True}
        if command=='launch':
            paths=shlex.split(rest)
            if len(paths)>1:raise ValueError('launch expects one quoted file path')
            return self.launch(paths[0] if paths else None)
        if command=='request':
            name,_,payload=rest.partition(' ')
            arguments=json.loads(payload or '{}')
            if not isinstance(arguments,dict):raise ValueError('DAP arguments must be a JSON object')
            return self.client.request(name,**arguments)
        if not self.launched:raise ValueError('Launch a config first')
        if command in ('continue','next','step','out'):
            self.client.request({'step':'stepIn','out':'stepOut'}.get(command,command),threadId=1)
            return self.stopped()
        if command=='where':return self.client.request('stackTrace',threadId=1)
        if command=='print':return self.client.request('evaluate',expression=rest,context='repl')
        if command=='tokens':return self.client.request('llmdbg/promptTokens')
        if command=='vector':return self.client.request('llmdbg/vector')
        if command=='lens':
            first=rest.split(maxsplit=1)[0] if rest else 'logit'
            if first not in ('j','r','jpp','logit') and not first.startswith('lens:'):
                return self.client.request('llmdbg/decodeExpression',expression=rest)
            words=shlex.split(rest);mode=words[0] if words else 'logit'
            if len(words)>2:raise ValueError('Use lens EXPRESSION or lens MODE VIEW')
            if mode in ('j','r','jpp'):mode='lens:'+mode
            return self.client.request('llmdbg/neuronLens',mode=mode,view=words[1] if len(words)>1 else 'residual-post')
        if command=='vars':
            if rest:return self.client.request('variables',variablesReference=int(rest))
            frames=self.client.request('stackTrace',threadId=1)['stackFrames']
            scopes=self.client.request('scopes',frameId=frames[0]['id'])['scopes']
            return {'scopes':[dict(name=s['name'],**self.client.request('variables',variablesReference=s['variablesReference'])) for s in scopes]}
        if command in ('break','clear'):
            if not self.source:raise ValueError('No active source path')
            proposed=[] if command=='clear' else list(self.breakpoints)
            if command=='break':
                location,_,condition=rest.partition(' ')
                lines=[int(location)] if location.isdigit() else self.calls.get(location)
                if not lines:raise ValueError('Use a source line number or authored call name (e.g. mlp)')
                if not isinstance(lines,(tuple,list)):lines=[lines]
                proposed.extend(dict(line=line,**({'condition':condition} if condition else {})) for line in lines)
            result=self.client.request('setBreakpoints',source={'path':self.source},breakpoints=proposed)
            if any(not bp.get('verified') for bp in result.get('breakpoints',[])):
                raise DAPError('Unverified breakpoint: '+json.dumps(result))
            self.breakpoints=proposed
            return result
        raise ValueError(f'Unknown command: {command}. Type help.')


def run(args):
    """Own a private host by default; never terminate an explicitly connected host."""
    owned=None;client=None
    try:
        if args.connect:
            host,separator,port=args.connect.rpartition(':')
            if not separator:raise ValueError('--connect expects HOST:PORT')
            address=(host,int(port))
        else:
            owned=subprocess.Popen([sys.executable,'-u','-m','llmdbg.host','--host','127.0.0.1','--port','0'],
                stdout=subprocess.PIPE,stderr=None,text=True)
            ready=queue.Queue()
            def consume():
                for line in owned.stdout:
                    if line.startswith('LLMDBG_HOST_READY '):ready.put(int(line.split()[-1]))
                    else:print(line,end='',file=sys.stderr)
                ready.put(None)
            reader=threading.Thread(target=consume,daemon=True);reader.start()
            port=ready.get(timeout=min(args.timeout,30))
            if port is None:raise DAPError('Private host exited during startup')
            address=('127.0.0.1',port)
        client=Client(address,args.timeout,output=lambda text:print(text,end='',file=sys.stderr,flush=True))
        client.request('initialize',adapterID='llmdbg',clientID='llmdbg-terminal',linesStartAt1=True,columnsStartAt1=True)
        terminal=Terminal(client,args)
        def emit(command,result=None,error=None):
            if args.json:
                print(json.dumps(dict(command=command,ok=error is None,**({'error':str(error)} if error else {'body':result})),ensure_ascii=False),flush=True)
            elif error:print('Error: '+str(error),file=sys.stderr)
            elif result.get('help'):print(result['help'])
            else:print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if args.config or args.demo:
            emit('launch',terminal.launch(args.config))
        def execute(line):
            if not line.strip() or line.lstrip().startswith('#'):return True
            try:
                result=terminal.execute(line);emit(line,result)
                return not result.get('quit',False)
            except (DAPError,ValueError,TimeoutError,EOFError) as error:
                emit(line,error=error)
                if args.batch or args.commands:raise SystemExit(1)
                return True
        for command in args.commands:
            if not execute(command):return
        if args.commands and not args.batch:return
        interactive=sys.stdin.isatty() and not args.batch
        if interactive:print('llmdbg terminal · type help for commands',file=sys.stderr)
        while True:
            try:line=input('llmdbg> ' if interactive and not args.json else '')
            except EOFError:break
            if not execute(line):break
    except (DAPError,ValueError,TimeoutError,EOFError,OSError,queue.Empty) as error:
        if args.json:print(json.dumps({'ok':False,'error':str(error)}),flush=True)
        else:print('Error: '+str(error),file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nInterrupted; disconnecting terminal client.',file=sys.stderr)
        raise SystemExit(130)
    finally:
        if client:
            client.timeout=min(client.timeout,2)
            try:client.request('disconnect')
            except (OSError,EOFError,TimeoutError,DAPError):pass
            client.close()
        if owned:
            owned.terminate()
            try:owned.wait(timeout=5)
            except subprocess.TimeoutExpired:owned.kill();owned.wait()
            if owned.stdout:owned.stdout.close()
