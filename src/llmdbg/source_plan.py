"""Source-derived, straight-line debugger plans; never execute user Python.

Model calls are capture adapters, not Python evaluation. Symbolic local values
retain assignment-time bindings, so aliases/reassignment cannot read future
locals or silently change the inputs to a captured model operation.
"""
from __future__ import annotations

import ast
from copy import deepcopy


ROOTS = {'attention', 'full_attention', 'gated_linear_attention', 'mlp'}


def registry(root, kind):
    mlp = root == 'mlp'
    gemma = kind in {'gemma2', 'gemma4'}
    mixed = 'gated_delta_mixing' if root == 'gated_linear_attention' else 'attention_mixing' if root == 'attention' and kind == 'gpt2' else 'full_attention_mixing'
    projected = 'mlp_projection' if mlp else 'attention_projection'
    raw = 'attention_projection_raw' if kind == 'gemma4' and not mlp else projected
    result = {
        'layernorm': (('input',), 'normalized_input'),
        'input_layernorm': (('input',), 'normalized_input'),
        'pre_feedforward_layernorm': (('input',), 'normalized_input'),
        'post_attention_layernorm': ((raw if not mlp else 'input',), projected if not mlp else 'normalized_input'),
        'qkv_projection': (('normalized_input',), 'attention_qkv'),
        'attention_mixing': (('attention_qkv',), 'attention_mixing'),
        'query_projection': (('normalized_input',), 'attention_query'),
        'key_projection': (('normalized_input',), 'attention_key'),
        'value_projection': (('normalized_input',), 'attention_value'),
        'cached_keys': ((('normalized_input' if kind == 'gemma4' else 'attention_key'),), 'attention_cached_keys'),
        'cached_values': ((('normalized_input' if kind == 'gemma4' else 'attention_value'),), 'attention_cached_values'),
        'attention_weights': (('attention_query','attention_cached_keys'), 'attention_weights'),
        'mix_values': (('attention_weights','attention_cached_values'), 'full_attention_mixing'),
        'full_attention': (('normalized_input',), 'full_attention_mixing'),
        'gated_delta': (('normalized_input',), 'gated_delta_mixing'),
        'convolved_qkv': (('normalized_input',), ('delta_query','delta_key','delta_value')),
        'gates': (('normalized_input',), ('delta_decay','delta_beta','delta_gate')),
        'cached_memory': ((), 'delta_previous_memory'),
        'update_memory': (('delta_previous_memory','delta_key','delta_value','delta_decay','delta_beta'), 'delta_updated_memory'),
        'read_memory': (('delta_query','delta_updated_memory'), 'delta_readout'),
        'gated_output_norm': (('delta_readout','delta_gate'), 'gated_delta_mixing'),
        'output_projection': ((mixed,), raw),
        'expansion': (('normalized_input',), 'mlp_expansion_pre_activation'),
        'gate_projection': (('normalized_input',), 'mlp_gate_pre_activation'),
        'up_projection': (('normalized_input',), 'mlp_up_projection'),
        'silu': (('mlp_gate_pre_activation',), 'mlp_gate_activation'),
        'gelu': ((('mlp_gate_pre_activation' if gemma else 'mlp_expansion_pre_activation'),), 'mlp_gate_activation' if gemma else 'mlp_expansion'),
        'projection': (('mlp_expansion',), 'mlp_projection'),
        'down_projection': (('mlp_expansion',), 'mlp_projection'),
        'post_feedforward_layernorm': (('mlp_projection',), 'mlp_projection'),
    }
    if mlp:
        allowed = ({'layernorm','expansion','gelu','projection'} if kind=='gpt2' else
                   {'pre_feedforward_layernorm','gate_projection','up_projection','gelu','down_projection','post_feedforward_layernorm'} if gemma else
                   {'post_attention_layernorm','gate_projection','up_projection','silu','down_projection'})
    elif root == 'gated_linear_attention':
        allowed = {'input_layernorm','gated_delta','convolved_qkv','gates','cached_memory','update_memory','read_memory','gated_output_norm','output_projection'}
    elif kind == 'gpt2':
        allowed = {'layernorm','qkv_projection','attention_mixing','output_projection'}
    else:
        allowed = {'input_layernorm','query_projection','key_projection','value_projection','cached_keys','cached_values','attention_weights','mix_values','full_attention','output_projection','post_attention_layernorm'}
    return {name: spec for name,spec in result.items() if name in allowed}


def compile_plans(tree, kind, roots):
    functions = {node.name:node for node in tree.body if isinstance(node, ast.FunctionDef)}
    return {root: Compiler(functions, root, kind).compile() for root in functions if root in roots and root in ROOTS}


class Compiler:
    def __init__(self, functions, root, kind):
        self.functions, self.root, self.kind = functions, root, kind
        self.adapters = registry(root,kind)
        self.steps, self.calls, self.frames, self.active = [], {}, [], []
        self.last_vector = 'input'
        self.seen = set()

    def fail(self,node,message):
        raise ValueError(f'line {node.lineno}: {message}')

    def emit(self,node,operation,vector=None,kind='operation'):
        if len(self.steps) >= 512:
            self.fail(node,'helper plan exceeds 512 source instructions')
        self.steps.append({'line':node.lineno,'operation':operation,'vector':vector or self.last_vector,
                           'kind':kind,'frames':deepcopy(self.frames)})
        if kind == 'operation':
            self.last_vector = vector

    def expression(self,node,env):
        compiler = self
        class Bind(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id == 'W_o':
                    return node
                if node.id not in env:
                    compiler.fail(node,f'local {node.id!r} is not defined at this line')
                return ast.copy_location(ast.parse(env[node.id],mode='eval').body,node)
            def visit_Call(self,node):
                if (not isinstance(node.func,ast.Attribute) or node.func.attr not in {'size','dim','numel'}
                        or node.keywords):
                    compiler.fail(node,'only model adapters, helper calls, and read-only tensor size/dim/numel calls are supported')
                return self.generic_visit(node)
            def visit_Attribute(self,node):
                if node.attr not in {'shape','dtype','device','ndim','requires_grad','size','dim','numel'}:
                    compiler.fail(node,f'unsupported attribute {node.attr!r}')
                return self.generic_visit(node)
        allowed = (ast.Name,ast.Constant,ast.BinOp,ast.UnaryOp,ast.Subscript,ast.Slice,ast.Tuple,ast.List,
                   ast.Load,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.MatMult,ast.USub,ast.UAdd,ast.Attribute,ast.Call)
        if any(not isinstance(part,allowed) for part in ast.walk(node)):
            self.fail(node,'unsupported tensor expression; use names, slices, tuples, and + - * / @')
        if any(isinstance(part,ast.Constant) and type(part.value) not in {int,float,type(None)} for part in ast.walk(node)):
            self.fail(node,'only numeric constants are supported in helper expressions')
        value = ast.unparse(Bind().visit(deepcopy(node)))
        if len(value) > 16384:
            self.fail(node,'expanded helper expression is too large')
        return value

    def assign(self,target,value,env):
        if isinstance(target,ast.Name):
            env[target.id] = value
        elif isinstance(target,(ast.Tuple,ast.List)):
            expr = ast.parse(value,mode='eval').body
            if not isinstance(expr,(ast.Tuple,ast.List)) or len(expr.elts) != len(target.elts):
                self.fail(target,'tuple assignment requires the same number of values and targets')
            for child,item in zip(target.elts,expr.elts):
                self.assign(child,ast.unparse(item),env)
        else:
            self.fail(target,'only local-name assignment and tuple unpacking are supported; tensor mutation is not')

    def value(self,node,env):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in self.functions:
            args = [self.expression(arg,env) for arg in node.args]
            kwargs = {arg.arg:self.expression(arg.value,env) for arg in node.keywords}
            if None in kwargs:
                self.fail(node,'expanded helper keyword arguments are not supported')
            call = len(self.steps)
            self.emit(node,node.func.id,kind='call')
            value = self.function(node.func.id,args,kwargs,node)
            self.calls[call] = {'name':node.func.id,'call':call,'start':call+1,'end':len(self.steps)-1}
            return value
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and env.get(node.func.value.id) == '__block__':
            operation = node.func.attr
            if operation not in self.adapters or node.keywords:
                self.fail(node,f'unsupported model operation {operation!r}; a capture adapter is required')
            inputs, output = self.adapters[operation]
            args = tuple(self.expression(arg,env) for arg in node.args)
            if args != inputs:
                self.fail(node,f'{operation} inputs differ from the captured native computation; expected {inputs}, got {args}. Changed model computation needs a backend adapter')
            if operation in self.seen:
                self.fail(node,f'repeated model operation {operation!r} needs occurrence-specific capture support')
            self.seen.add(operation)
            vector = 'delta_qkv' if operation == 'convolved_qkv' else 'delta_beta' if operation == 'gates' else output
            self.emit(node,operation,vector)
            return '('+', '.join(output)+')' if isinstance(output,tuple) else output
        expression = self.expression(node,env)
        residual = 'mlp_projection' if self.root == 'mlp' else 'attention_projection'
        native = {'input + '+residual:'residual_add', residual+' + input':'residual_add',
                  'mlp_gate_activation * mlp_up_projection':'gated_multiply',
                  'mlp_up_projection * mlp_gate_activation':'gated_multiply'}.get(expression)
        if native:
            vector = 'output' if native == 'residual_add' else 'mlp_expansion'
            if native not in self.seen:
                self.seen.add(native)
                self.emit(node,native,vector)
            else:
                self.emit(node,'assign',kind='expression')
            return vector
        self.emit(node,'assign',kind='expression')
        return expression

    def function(self,name,args,kwargs,call_node):
        node = self.functions[name]
        if name in self.active or len(self.active) >= 16:
            self.fail(call_node,'recursive helpers or helper depth above 16 are not supported')
        if node.args.defaults or node.args.kw_defaults or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            self.fail(node,'helpers require fixed parameters without defaults or variadic arguments')
        parameters = [arg.arg for arg in (*node.args.posonlyargs,*node.args.args)]
        if len(args)>len(parameters) or set(kwargs)-set(parameters) or set(parameters[:len(args)]) & set(kwargs):
            self.fail(call_node,f'invalid arguments for helper {name}')
        env = dict(zip(parameters,args)); env.update(kwargs)
        if set(env) != set(parameters):
            self.fail(call_node,f'missing arguments for helper {name}')
        local_names = sorted({part.id for statement in node.body for part in ast.walk(statement)
                              if isinstance(part,ast.Name) and isinstance(part.ctx,ast.Store)} | set(parameters))
        self.active.append(name)
        frame = {'name':name,'callLine':call_node.lineno,'bindings':env,'localNames':local_names}
        self.frames.append(frame)
        returned = None
        for index,statement in enumerate(node.body):
            if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Constant) and isinstance(statement.value.value,str):
                continue  # docstring, not an executable stop
            if isinstance(statement,(ast.Assign,ast.AnnAssign)) and statement.value is not None:
                first = len(self.steps)
                value = self.value(statement.value,env)
                self.statement_line(first,statement.lineno)
                targets = statement.targets if isinstance(statement,ast.Assign) else [statement.target]
                for target in targets:
                    self.assign(target,value,env)
            elif isinstance(statement,ast.Return) and statement.value is not None:
                if index != len(node.body)-1:
                    self.fail(statement,'unreachable statements after return are not supported')
                before = len(self.steps)
                returned = self.value(statement.value,env)
                self.statement_line(before,statement.lineno)
                if isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id in self.functions:
                    self.emit(statement,'return',kind='return')
                elif self.steps[-1]['kind']=='expression':
                    self.steps[-1].update(operation='return',kind='return')
                elif len(self.steps)==before:
                    self.emit(statement,'return',kind='return')
                break
            else:
                self.fail(statement,'only straight-line assignments, helper calls, and returns are supported inside helpers')
        if returned is None:
            self.fail(node,f'helper {name} must return a value')
        self.frames.pop(); self.active.pop()
        return returned

    def statement_line(self,first,line):
        """Multiline RHS calls still stop on their assignment/return statement."""
        self.steps[first]['line'] = line
        call = self.calls.get(first)
        if call:
            parent_depth = len(self.frames)
            for step in self.steps[first+1:call['end']+1]:
                if len(step['frames']) > parent_depth:
                    step['frames'][parent_depth]['callLine'] = line

    def compile(self):
        node = self.functions[self.root]
        returned = self.function(self.root,['input','__block__'],{},node)
        if returned != 'output':
            self.fail(node,'model helper must return its captured residual output; altered return computation needs backend support')
        if self.kind in {'gemma2','gemma4'}:
            required = 'post_feedforward_layernorm' if self.root=='mlp' else 'post_attention_layernorm'
            if required not in self.seen:
                self.fail(node,f'missing {required}; source must preserve the native output normalization')
        # Assignments share their statement's source line, not a multiline RHS.
        for step in self.steps:
            step['frames'] = [{**frame,'bindings':{k:v for k,v in frame['bindings'].items() if v!='__block__'}} for frame in step['frames']]
        return {'steps':self.steps,'calls':self.calls}
