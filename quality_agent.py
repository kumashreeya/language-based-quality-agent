from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import ast, math, re, json, subprocess, tempfile, os, sys, requests
from rag_guidelines import get_guidelines_for_language, get_rules_for_evaluation

def call_llm(prompt, system="", max_tokens=500):
    payload = {"model":"qwen2.5-coder:7b","prompt":f"{system}\n\n{prompt}" if system else prompt,"stream":False,"options":{"num_predict":max_tokens,"temperature":0}}
    response = requests.post("http://localhost:11434/api/generate",json=payload,timeout=180)
    result = response.json().get("response","")
    if "<think>" in result and "</think>" in result: result = result.split("</think>")[-1].strip()
    return result

def tool_loc(code):
    lines=code.splitlines();total=len(lines);blank=sum(1 for l in lines if l.strip()=="")
    comment=sum(1 for l in lines if l.strip().startswith(("#","//","/*","*","--")))
    lloc=total-blank-comment;ratio=comment/max(total,1)*100
    return f"LOC:\n  SLOC: {total}\n  LLOC: {lloc}\n  Blank: {blank}\n  Comments: {comment}\n  Comment ratio: {ratio:.1f}%"

def tool_token_count(code):
    return f"Token Count:\n  Chars: {len(code)}\n  Words: {len(code.split())}\n  Lines: {len(code.splitlines())}"

def tool_redundancy(code):
    lines=[l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith(("#","//","/*","*"))]
    if len(lines)<3: return "Redundancy: Too short."
    blocks={}
    for i in range(len(lines)-2):
        block="\n".join(lines[i:i+3]);blocks.setdefault(block,[]).append(i+1)
    dups={k:v for k,v in blocks.items() if len(v)>1}
    dup_lines=sum(3*(len(v)-1) for v in dups.values());density=dup_lines/max(len(lines),1)*100
    return f"Redundancy:\n  Duplicates: {len(dups)}\n  Density: {density:.1f}%"

def tool_line_length(code):
    lines=code.splitlines()
    if not lines: return "Line Length: No code."
    lengths=[len(l) for l in lines];over80=sum(1 for l in lengths if l>80);over100=sum(1 for l in lengths if l>100)
    violations=[f"  Line {i+1}: {len(l)} chars" for i,l in enumerate(lines) if len(l)>100]
    out=f"Line Length:\n  Max: {max(lengths)}\n  Avg: {sum(lengths)//len(lengths)}\n  >80: {over80}, >100: {over100}"
    if violations: out+="\n"+"\n".join(violations[:5])
    return out

def tool_func_count(code):
    patterns=[r'^\s*def\s+\w+',r'^\s*\w[\w\s\*&:<>]*\s+\w+\s*\([^)]*\)\s*\{?',r'^\s*(public|private|protected)[\w\s]*\s+\w+\s*\(',r'function\s+\w+']
    total=sum(len(re.findall(p,code,re.MULTILINE)) for p in patterns)
    return f"Function Count:\n  Detected: {max(total,0)}"

def tool_nesting(code):
    max_indent=0;deep=[]
    for i,line in enumerate(code.splitlines()):
        s=line.lstrip()
        if not s: continue
        indent=len(line)-len(s);level=indent//4 if '\t' not in line else indent
        if level>max_indent: max_indent=level
        if level>=4: deep.append(f"  Line {i+1}: depth={level}")
    rating="Good" if max_indent<=3 else ("Moderate" if max_indent<=5 else "DEEP")
    out=f"Nesting:\n  Max depth: {max_indent}\n  Rating: {rating}"
    if deep: out+="\n"+"\n".join(deep[:5])
    return out

def tool_comments(code):
    lines=code.splitlines();total=len(lines)
    comments=[i+1 for i,l in enumerate(lines) if l.strip().startswith(("#","//","/*","*","--"))]
    ratio=len(comments)/max(total,1)*100;has_header=any(i<=3 for i in comments)
    out=f"Comments:\n  Count: {len(comments)}\n  Ratio: {ratio:.1f}%\n  Header: {'Yes' if has_header else 'MISSING'}"
    if ratio<5: out+="\n  WARNING: Very few comments"
    return out

def tool_cc(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "CC: Skipped (not Python)"
    results=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            cc=1
            for child in ast.walk(node):
                if isinstance(child,(ast.If,ast.While,ast.For,ast.ExceptHandler)): cc+=1
                elif isinstance(child,ast.BoolOp): cc+=len(child.values)-1
            results.append(f"  {node.name}(): CC={cc} ({'low' if cc<=10 else ('moderate' if cc<=20 else 'HIGH')})")
    return "Cyclomatic Complexity:\n"+("\n".join(results) if results else "  No functions.")

def tool_cogc(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "CogC: Skipped (not Python)"
    results=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            cogc=[0]
            def calc(n,nesting=0):
                inc=0
                if isinstance(n,(ast.If,ast.While,ast.For)): inc=1+nesting
                elif isinstance(n,ast.ExceptHandler): inc=1+nesting
                elif isinstance(n,ast.BoolOp): inc=1
                cogc[0]+=inc;cn=nesting+(1 if isinstance(n,(ast.If,ast.While,ast.For,ast.With,ast.Try)) else 0)
                for c in ast.iter_child_nodes(n): calc(c,cn)
            calc(node);results.append(f"  {node.name}(): CogC={cogc[0]}{' <- HIGH' if cogc[0]>15 else ''}")
    return "Cognitive Complexity:\n"+("\n".join(results) if results else "  No functions.")

def tool_halstead(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "Halstead: Skipped (not Python)"
    ops=set();opds=set();N1=N2=0
    for node in ast.walk(tree):
        if isinstance(node,(ast.BinOp,ast.UnaryOp)): ops.add(type(node.op).__name__);N1+=1
        elif isinstance(node,ast.Compare):
            for op in node.ops: ops.add(type(op).__name__);N1+=1
        elif isinstance(node,ast.BoolOp): ops.add(type(node.op).__name__);N1+=1
        elif isinstance(node,ast.Name): opds.add(node.id);N2+=1
        elif isinstance(node,ast.Constant): opds.add(str(node.value));N2+=1
    vocab=len(ops)+len(opds)
    if vocab==0: return "Halstead: Not enough data."
    vol=(N1+N2)*math.log2(max(vocab,1));diff=(len(ops)/2)*(N2/max(len(opds),1))
    return f"Halstead:\n  Volume: {vol:.1f}\n  Difficulty: {diff:.1f}\n  Est. bugs: {vol/3000:.2f}"

def tool_mi(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "MI: Skipped (not Python)"
    lines=code.splitlines();lloc=sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    cc=1
    for node in ast.walk(tree):
        if isinstance(node,(ast.If,ast.While,ast.For,ast.ExceptHandler)): cc+=1
        elif isinstance(node,ast.BoolOp): cc+=len(node.values)-1
    ops=set();opds=set();N1=N2=0
    for node in ast.walk(tree):
        if isinstance(node,(ast.BinOp,ast.UnaryOp,ast.BoolOp)): ops.add(type(getattr(node,'op',node)).__name__);N1+=1
        elif isinstance(node,ast.Compare):
            for op in node.ops: ops.add(type(op).__name__);N1+=1
        elif isinstance(node,ast.Name): opds.add(node.id);N2+=1
        elif isinstance(node,ast.Constant): opds.add(str(node.value));N2+=1
    vol=(N1+N2)*math.log2(max(len(ops)+len(opds),2))
    mi=171-5.2*math.log(max(vol,1))-0.23*cc-16.2*math.log(max(lloc,1))
    mi_s=max(0,mi*100/171)
    return f"MI:\n  Score: {mi_s:.1f}/100\n  Rating: {'Good' if mi_s>=85 else ('Moderate' if mi_s>=65 else 'POOR')}"

def tool_docstring(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "Docstring: Skipped (not Python)"
    total=documented=0;missing=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            total+=1
            has_doc=(node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str))
            if has_doc: documented+=1
            else: missing.append(f"  [MISSING] '{node.name}' line {node.lineno}")
    if total==0: return "Docstring: No functions/classes."
    out=f"Docstring Coverage:\n  {documented}/{total} ({documented/total*100:.0f}%)"
    if missing: out+="\n"+"\n".join(missing[:10])
    return out

def tool_type_hints(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "Type Hints: Skipped (not Python)"
    tp=ap=tr=ar=0
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            tr+=1
            if node.returns: ar+=1
            for arg in node.args.args:
                if arg.arg in ('self','cls'): continue
                tp+=1
                if arg.annotation: ap+=1
    total=tp+tr;annotated=ap+ar
    if total==0: return "Type Hints: No functions."
    return f"Type Hints:\n  Params: {ap}/{tp}, Returns: {ar}/{tr}\n  Overall: {annotated/total*100:.0f}%"

def tool_dead_code(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "Dead Code: Skipped (not Python)"
    defined=set();used=set()
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): defined.add(node.name)
        elif isinstance(node,ast.ClassDef): defined.add(node.name)
        elif isinstance(node,ast.Name):
            if isinstance(node.ctx,ast.Store): defined.add(node.id)
            elif isinstance(node.ctx,ast.Load): used.add(node.id)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            for alias in node.names: defined.add(alias.asname or alias.name)
    ignore={'print','len','range','int','str','float','list','dict','set','tuple','True','False','None','self','super','__name__','__main__','open','type','isinstance','input','exit','max','min','sorted','enumerate','zip','map','filter','any','all','abs','round','sum','hasattr','getattr'}
    issues=[f"  [UNUSED] '{n}'" for n in sorted(defined-used-ignore)]
    if issues: return f"Dead Code ({len(issues)}):\n"+"\n".join(issues)
    return "Dead Code: None."

def tool_bug_density(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "Bug Density: Skipped (not Python)"
    smells=0;smell_list=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            bl=(node.end_lineno-node.lineno+1) if hasattr(node,'end_lineno') else 0
            if bl>50: smells+=1;smell_list.append(f"  Long: {node.name}()={bl} lines")
        elif isinstance(node,ast.ExceptHandler):
            if node.type is None: smells+=1;smell_list.append(f"  Bare except line {node.lineno}")
        elif isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name) and node.func.id in ('eval','exec'): smells+=1;smell_list.append(f"  Dangerous: {node.func.id}() line {node.lineno}")
    out=f"Bug Density:\n  Smells: {smells}"
    if smell_list: out+="\n"+"\n".join(smell_list)
    return out

def tool_imports(code):
    import importlib
    try: tree=ast.parse(code)
    except SyntaxError: return "Imports: Skipped (not Python)"
    issues=[];valid=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names:
                try: importlib.import_module(alias.name);valid+=1
                except ImportError: issues.append(f"  '{alias.name}' NOT FOUND")
        elif isinstance(node,ast.ImportFrom) and node.module:
            try: importlib.import_module(node.module);valid+=1
            except ImportError: issues.append(f"  '{node.module}' NOT FOUND")
    out=f"Imports:\n  Valid: {valid}\n  Hallucinated: {len(issues)}"
    if issues: out+="\n"+"\n".join(issues)
    return out

def detect_lang(code):
    indicators={"python":["def ","import ","elif ","self.","print("],"javascript":["function ","const ","let ","=>","console.log"],"java":["public class","public static","System.out","void "],"cpp":["#include","std::","cout","nullptr","template<"],"c":["#include","printf","scanf","malloc","free("],"csharp":["namespace ","using System","Console."],"go":["func ","package ","fmt.",":= "],"rust":["fn ","let mut","impl ","pub fn"],"r":["<-","library(","data.frame","ggplot"],"sql":["SELECT ","FROM ","WHERE ","INSERT ","CREATE TABLE"],"visualbasic":["Sub ","Dim ","End Sub","Module "],"delphi":["procedure ","begin","end;","uses "]}
    scores={lang:sum(1 for p in patterns if p in code) for lang,patterns in indicators.items()}
    if scores.get("c",0)>0 and scores.get("cpp",0)>0:
        if "class " in code or "cout" in code or "std::" in code: scores["c"]=0
        else: scores["cpp"]=0
    return max(scores,key=scores.get) if max(scores.values())>0 else "unknown"

UNIVERSAL_TOOLS={"loc":{"fn":tool_loc,"desc":"Lines of code"},"token_count":{"fn":tool_token_count,"desc":"Char/word/line count"},"redundancy":{"fn":tool_redundancy,"desc":"Code duplication"},"line_length":{"fn":tool_line_length,"desc":"Line length check"},"func_count":{"fn":tool_func_count,"desc":"Function count"},"nesting":{"fn":tool_nesting,"desc":"Nesting depth"},"comments":{"fn":tool_comments,"desc":"Comment quality"}}
PYTHON_TOOLS={"cc":{"fn":tool_cc,"desc":"Cyclomatic complexity"},"cogc":{"fn":tool_cogc,"desc":"Cognitive complexity"},"halstead":{"fn":tool_halstead,"desc":"Halstead metrics"},"mi":{"fn":tool_mi,"desc":"Maintainability Index"},"docstring":{"fn":tool_docstring,"desc":"Docstring coverage"},"type_hints":{"fn":tool_type_hints,"desc":"Type hint coverage"},"dead_code":{"fn":tool_dead_code,"desc":"Dead code"},"bug_density":{"fn":tool_bug_density,"desc":"Bug density"},"imports":{"fn":tool_imports,"desc":"Import check"}}
ALL_TOOLS={**UNIVERSAL_TOOLS,**PYTHON_TOOLS}

def read_code_file(filepath):
    filepath=filepath.strip().strip("'\"")
    if not os.path.exists(filepath): return None,f"File not found: {filepath}"
    try:
        with open(filepath,"r",encoding="utf-8",errors="replace") as f: code=f.read()
        return code,os.path.basename(filepath)
    except Exception as e: return None,str(e)

class AgentState(TypedDict):
    code:str;mode:str;user_request:str;detected_language:str;guidelines_context:str;guideline_checks:str;tools_to_run:list;tool_results:str;interpretation:str;final_report:str;loop_count:int;max_loops:int

def node_detect(state):
    lang=detect_lang(state["code"]);print(f"\n[Step 1] Language: {lang}")
    return {**state,"detected_language":lang}

def node_guidelines(state):
    lang=state["detected_language"];print(f"[Step 2] Querying RAG for {lang} guidelines...")
    guidelines=get_guidelines_for_language(lang,top_k=8)
    if guidelines: print(f"  Found {lang} guidelines")
    else: print(f"  No {lang} guidelines in RAG")
    return {**state,"guidelines_context":guidelines or ""}

def node_pick_tools(state):
    lang=state["detected_language"];is_python=(lang=="python");guidelines=state["guidelines_context"]
    print(f"[Step 3] Brain deciding what to check...")
    if state["mode"]=="manual":
        available=ALL_TOOLS if is_python else UNIVERSAL_TOOLS
        tool_list="\n".join([f'  "{n}": {i["desc"]}' for n,i in available.items()])
        system=f"Pick tools for: {state['user_request']}\nAvailable:\n{tool_list}\nRespond ONLY JSON array."
        raw=call_llm(state['user_request'],system,200)
        try:
            match=re.search(r'\[.*?\]',raw,re.DOTALL);tools=json.loads(match.group()) if match else []
        except: tools=[]
        valid=[t for t in tools if t in available]
        if not valid: valid=list(available.keys())
        return {**state,"tools_to_run":valid,"guideline_checks":""}
    available=ALL_TOOLS if is_python else UNIVERSAL_TOOLS
    tool_list="\n".join([f'  "{n}": {i["desc"]}' for n,i in available.items()])
    if guidelines:
        system=f"""You are a quality agent for {lang} code.
TOOLS (ONLY pick from this list):
{tool_list}

OFFICIAL {lang.upper()} GUIDELINES:
{guidelines[:2500]}

Based on guidelines, pick ONLY relevant tools and list guideline checks.
RESPOND ONLY JSON:
{{"tools":["tool1","tool2"],"guideline_checks":["Check: specific rule","Check: another rule"],"reasoning":"why"}}"""
    else:
        system=f"You are a quality agent for {lang}.\nTOOLS:\n{tool_list}\nPick relevant tools.\nRESPOND ONLY JSON:\n{{\"tools\":[\"tool1\"],\"guideline_checks\":[\"Check: naming\"],\"reasoning\":\"why\"}}"
    raw=call_llm(f"What to check in this {lang} code:\n```\n{state['code'][:2000]}\n```",system,600)
    tools=[];guideline_checks=""
    try:
        match=re.search(r'\{.*\}',raw,re.DOTALL)
        if match:
            data=json.loads(match.group());tools=[t for t in data.get("tools",[]) if t in available]
            checks=data.get("guideline_checks",[])
            if checks:
                guideline_checks="\n".join(f"  - {c}" for c in checks)
                print(f"  Brain requested {len(checks)} checks:")
                for c in checks[:5]: print(f"    {c}")
            reasoning=data.get("reasoning","")
            if reasoning: print(f"  Reasoning: {reasoning[:150]}")
    except: pass
    if not tools: tools=["loc","comments","nesting"]
    print(f"  Tools: {tools}")
    return {**state,"tools_to_run":tools,"guideline_checks":guideline_checks}

def node_run_tools(state):
    print(f"[Step 4] Running {len(state['tools_to_run'])} tools...")
    results=[]
    for name in state["tools_to_run"]:
        if name not in ALL_TOOLS: continue
        print(f"  {name}...");
        try: result=ALL_TOOLS[name]["fn"](state["code"])
        except Exception as e: result=f"{name} Error: {e}"
        results.append(result)
    return {**state,"tool_results":"\n\n".join(results)}

def node_evaluate(state):
    lang=state["detected_language"];is_python=(lang=="python");guidelines=state["guidelines_context"];guideline_checks=state["guideline_checks"]
    print(f"[Step 5] Evaluating {lang} code...")
    rules=get_rules_for_evaluation(lang,f"{lang} coding rules naming style documentation",top_k=5)
    rules_section=""
    if rules: rules_section=f"\n\nOFFICIAL {lang.upper()} GUIDELINES:\n{rules}\n"
    elif guidelines: rules_section=f"\n\nOFFICIAL {lang.upper()} GUIDELINES:\n{guidelines[:2000]}\n"
    checks_section=""
    if guideline_checks: checks_section=f"\nGUIDELINE CHECKS TO PERFORM:\n{guideline_checks}\n\nFor EACH check: state rule, examine code, give line numbers, PASS/WARNING/FAIL.\n"
    if is_python:
        system=f"You are a STRICT Python reviewer.\nInterpret tool results AND perform guideline checks.\n{rules_section}{checks_section}\nThresholds: CC>10=WARN CC>20=FAIL. MI<65=FAIL. 0% docstrings=FAIL.\nCITE guidelines. End with SUMMARY and VERDICT: PASS or FAIL."
    else:
        system=f"You are a STRICT {lang} reviewer.\nTools measured metrics. You MUST also perform guideline checks.\n{rules_section}{checks_section}\nSECTION 1: Interpret tool results. PASS/WARNING/FAIL.\nSECTION 2: For each guideline check, examine code, cite rule, give lines, PASS/WARNING/FAIL.\nEnd with SUMMARY and VERDICT: PASS or FAIL."
    prompt=f"Language: {lang}\n\nCode:\n```\n{state['code'][:3000]}\n```\n\nTool results:\n{state['tool_results']}\n\nEvaluate strictly."
    response=call_llm(prompt,system,1500)
    return {**state,"interpretation":response,"loop_count":state["loop_count"]+1}

def node_report(state):
    lang=state["detected_language"];has_rag="Yes" if state["guidelines_context"] else "No"
    checks_section=""
    if state["guideline_checks"]: checks_section=f"\nGuideline checks (brain-selected):\n{state['guideline_checks']}\n"
    report=f"\n{'='*60}\nQUALITY AGENT REPORT (Guidelines-Driven)\n{'='*60}\nLanguage: {lang}\nGuidelines from RAG: {has_rag}\nTools (brain-selected): {', '.join(state['tools_to_run'])}\n{checks_section}\n--- TOOL RESULTS ---\n\n{state['tool_results']}\n\n--- {lang.upper()} EVALUATION ---\n\n{state['interpretation']}\n\n{'='*60}\n"
    return {**state,"final_report":report}

def should_continue(state): return "node_report"

def build_agent():
    graph=StateGraph(AgentState)
    for name,fn in [("node_detect",node_detect),("node_guidelines",node_guidelines),("node_pick_tools",node_pick_tools),("node_run_tools",node_run_tools),("node_evaluate",node_evaluate),("node_report",node_report)]:
        graph.add_node(name,fn)
    graph.add_edge(START,"node_detect");graph.add_edge("node_detect","node_guidelines");graph.add_edge("node_guidelines","node_pick_tools");graph.add_edge("node_pick_tools","node_run_tools");graph.add_edge("node_run_tools","node_evaluate");graph.add_conditional_edges("node_evaluate",should_continue);graph.add_edge("node_report",END)
    return graph.compile()

def main():
    agent=build_agent()
    print("="*60);print("QUALITY AGENT v7 (Guidelines-Driven | RAG Decides)");print("="*60)
    print("\nPipeline: Detect -> Query RAG -> Brain picks tools+checks -> Run -> Evaluate -> Report")
    print(f"\n  Python: up to {len(ALL_TOOLS)} tools | Others: up to {len(UNIVERSAL_TOOLS)} tools + RAG checks")
    print("\nInput: paste code (then END) OR type a file path")
    print("LLM: Ollama qwen2.5-coder:7b")
    while True:
        print("\n"+"-"*60);print("Paste code (then END), OR file path, OR quit");print("-"*60)
        first_line=input().strip()
        if first_line.lower()=="quit": print("Goodbye!");return
        code=None;filename=None
        test_path=first_line.strip("'\"")
        if os.path.exists(test_path):
            code,filename=read_code_file(first_line)
            if code: print(f"\nLoaded: {filename} ({len(code.splitlines())} lines)")
            else: print(f"Error: {filename}");continue
        else:
            lines=[first_line]
            while True:
                try: line=input()
                except EOFError: break
                if line.strip()=="END": break
                if line.strip()=="quit": print("Goodbye!");return
                lines.append(line)
            code="\n".join(lines)
            if not code.strip(): print("No code.");continue
            print(f"\n{len(lines)} lines received.")
        req=input("ENTER=auto, or type request: ").strip()
        if req=="quit": return
        final=agent.invoke({"code":code,"mode":"auto" if not req else "manual","user_request":req,"detected_language":"","guidelines_context":"","guideline_checks":"","tools_to_run":[],"tool_results":"","interpretation":"","final_report":"","loop_count":0,"max_loops":3})
        print(final["final_report"])

if __name__=="__main__":
    main()
