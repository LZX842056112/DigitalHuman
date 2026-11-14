import React,{useEffect,useRef,useState} from 'react';
import {ArrowLeft,Loader2,Upload,UserRound} from 'lucide-react';
import {api,fileUrl,json} from '../api';
import {checkImage} from '../image-rules';
import EditPreview from './EditPreview';

type Variant={id:string;label:string;photo:string;portrait?:{folder:string;width?:number;height?:number;fps?:number}|null};
type Analysis={size:[number,number];box:[number,number,number,number];head:number};
type Plan={canvas:[number,number];skipped:string|null;gain:number};
type PlanResponse={plan:Plan;compose:{url:string|null;width:number;height:number};warnings:string[]};
type PromptPair={zh:string;en:string};
type PromptResponse={action:PromptPair;tone:PromptPair;expression:PromptPair};
type EditJob={id:string;status:string;stage:string;progress:number;error?:string;
 preview?:{version:number;file:string};
 result?:{variant_id?:string;label?:string;quality?:string;skipped?:string;metrics?:{capped?:boolean}}};

type Props={
 avatarId:string;
 variantId:string;
 variants:Variant[];
 locked:boolean;
 onClose:()=>void;
 onBusy:(busy:boolean)=>void;
 onDone:(variantId:string)=>void;
 onError:(message:string)=>void;
};

const ASPECTS:[string,string,string][]=[['3:4','3:4','竖版全身'],['9:16','9:16','手机竖屏']];
const stageText:Record<string,string>={queued:'等待处理',analyze:'分析人物',canvas:'准备画布',diffuse:'生成画面',compose:'拼接接缝',verify:'检查人脸',complete:'生成完成'};
const FACE_HINT='全身照会让对话画面里的人脸变小，建议同时保留一张半身造型。';

export default function EditWorkbench({avatarId,variantId,variants,locked,onClose,onBusy,onDone,onError}:Props){
 const [analysis,setAnalysis]=useState<Analysis|null>(null);
 const [analyzing,setAnalyzing]=useState(true);
 const [aspect,setAspect]=useState('3:4');
 const [action,setAction]=useState('');
 const [toneText,setToneText]=useState('');
 const [expression,setExpression]=useState('');
 const [prompts,setPrompts]=useState<PromptResponse|null>(null);
 const [plan,setPlan]=useState<Plan|null>(null);
 const [composeUrl,setComposeUrl]=useState<string|null>(null);
 const [warnings,setWarnings]=useState<string[]>([]);
 const [garment,setGarment]=useState<File|null>(null);
 const [garmentUrl,setGarmentUrl]=useState('');
 const [wearPrompt,setWearPrompt]=useState('');
 const [job,setJob]=useState<EditJob|null>(null);
 const [running,setRunning]=useState<'extend'|'wear'|null>(null);
 const [previewUrl,setPreviewUrl]=useState<string|null>(null);
 const [resultId,setResultId]=useState('');
 const version=useRef(0),alive=useRef(true);

 useEffect(()=>{alive.current=true;return()=>{alive.current=false}},[]);

 useEffect(()=>{
  let active=true;
  setAnalysis(null);setAnalyzing(true);setPlan(null);setComposeUrl(null);setWarnings([]);
  setJob(null);setPreviewUrl(null);setResultId('');version.current=0;
  void api('/avatars/'+encodeURIComponent(avatarId)+'/edit/analyze',{method:'POST',body:'{}',headers:{'Content-Type':'application/json'}})
   .then((value:Analysis)=>{if(active)setAnalysis(value)})
   .catch((error:Error)=>{if(active)onError(error.message)})
   .finally(()=>{if(active)setAnalyzing(false)});
  return()=>{active=false};
 },[avatarId,variantId]);

 useEffect(()=>{
  if(!analysis)return;
  let active=true;
  void api('/avatars/'+encodeURIComponent(avatarId)+'/edit/plan',{method:'POST',...json({variant_id:variantId,aspect})})
   .then((value:PlanResponse)=>{if(!active)return;setPlan(value.plan);setComposeUrl(value.compose.url);setWarnings(value.warnings||[])})
   .catch((error:Error)=>{if(active)onError(error.message)});
  return()=>{active=false};
 },[analysis,aspect,avatarId,variantId]);

 useEffect(()=>{
  if(!action.trim()&&!toneText.trim()&&!expression.trim()){setPrompts(null);return}
  const timer=setTimeout(()=>{
   void api('/avatars/'+encodeURIComponent(avatarId)+'/edit/prompt',{method:'POST',...json({action,tone:toneText,expression})})
    .then((value:PromptResponse)=>{if(alive.current)setPrompts(value)})
    .catch(()=>{});
  },300);
  return()=>clearTimeout(timer);
 },[avatarId,action,toneText,expression]);

 useEffect(()=>{
  if(!garment){setGarmentUrl('');return}
  const url=URL.createObjectURL(garment);setGarmentUrl(url);
  return()=>URL.revokeObjectURL(url);
 },[garment]);

 useEffect(()=>{
  if(!job||!running)return;
  let active=true,pending=false;
  const timer=setInterval(async()=>{
   if(pending)return;pending=true;
   try{
    const next:EditJob=await api('/jobs/'+encodeURIComponent(job.id));
    if(!active)return;
    setJob(next);
    const stamp=next.preview?.version||0;
    if(stamp&&stamp!==version.current){
     const url='/api/jobs/'+encodeURIComponent(next.id)+'/preview?v='+stamp;
     const image=new Image();
     image.onload=()=>{if(active){setPreviewUrl(url);version.current=stamp}};
     image.src=url;
    }
    if(!['queued','running'].includes(next.status)){
     setRunning(null);onBusy(false);
     if(next.result?.variant_id){setResultId(next.result.variant_id);onDone(next.result.variant_id)}
    }
   }catch(error){if(active)onError((error as Error).message)}
   finally{pending=false}
  },700);
  return()=>{active=false;clearInterval(timer)};
 },[job?.id,running]);

 async function startExtend(){
  setRunning('extend');setPreviewUrl(null);setResultId('');version.current=0;onBusy(true);
  try{
   const created=await api('/avatars/'+encodeURIComponent(avatarId)+'/extend',
    {method:'POST',...json({variant_id:variantId,aspect,action,tone:toneText,expression})});
   if(alive.current)setJob({id:created.job_id,status:'queued',stage:'queued',progress:0});
  }catch(error){setRunning(null);onBusy(false);onError((error as Error).message)}
 }

 async function startWear(){
  if(!garment)return;
  const problem=await checkImage(garment,'garment');
  if(problem){onError(problem);return}
  setRunning('wear');setPreviewUrl(null);setResultId('');version.current=0;onBusy(true);
  const data=new FormData();
  data.set('garment',garment);data.set('variant_id',variantId);
  if(wearPrompt.trim())data.set('prompt',wearPrompt.trim());
  try{
   const created=await api('/avatars/'+encodeURIComponent(avatarId)+'/outfit/wear/job',{method:'POST',body:data});
   if(alive.current)setJob({id:created.job_id,status:'queued',stage:'canvas',progress:0});
  }catch(error){setRunning(null);onBusy(false);onError((error as Error).message)}
 }

 const current=variants.find(item=>item.id===variantId)||variants[0];
 const sourceUrl=current?fileUrl(avatarId,current.photo):null;
 const resultVariant=variants.find(item=>item.id===resultId);
 const resultUrl=resultVariant?fileUrl(avatarId,resultVariant.photo):'';
 const busy=!!running;
 const extendReason=busy?'正在生成，请稍候':locked?'请先停止准备任务或结束对话':analyzing||!analysis?'请先等待照片分析完成':!plan?'正在计算画布':plan.skipped?'这张照片已接近全身，无需扩展':plan.gain<=0?'这张照片不需要扩展':'';
 const wearReason=busy?'正在生成，请稍候':locked?'请先停止准备任务或结束对话':!garment?'请先上传一张服装图片':'';
 const translateHint=(pair?:PromptPair,label?:string)=>{
  if(!pair||!pair.zh)return '';
  return pair.en?(label||'')+pair.en:'未翻译（质量可能下降）';
 };
 return <div className="photo-edit">
  <header className="photo-edit-head">
   <button type="button" className="photo-edit-back" disabled={busy} onClick={onClose}><ArrowLeft size={15}/>返回对话视图</button>
   <div><b>照片编辑</b><small>{analyzing?'正在分析人物…':plan?.skipped?'这张照片已接近全身':'可扩成全身照，也可按图换装'}</small></div>
  </header>
  <EditPreview sourceUrl={sourceUrl} composeUrl={composeUrl} previewUrl={previewUrl}
   resultUrl={resultUrl} progress={job?.progress||0}
   stage={job?stageText[job.stage]||'':''} hint={FACE_HINT}/>
  <div className="photo-edit-controls">
   <section className="photo-edit-group">
    <h3>画幅</h3>
    <div className="photo-segments">{ASPECTS.map(([value,label,note])=><button key={value} type="button"
      className={aspect===value?'on':''} disabled={busy||locked} title={note}
      onClick={()=>setAspect(value)}>{label}</button>)}</div>
   </section>
   <section className="photo-edit-group">
    <h3>提示词</h3>
    <label className="photo-edit-field"><span>动作</span>
     <input value={action} maxLength={40} disabled={busy||locked} placeholder="如：走向镜头，双手插兜"
      onChange={event=>setAction(event.target.value)}/></label>
    <label className="photo-edit-field"><span>语气</span>
     <input value={toneText} maxLength={40} disabled={busy||locked} placeholder="如：温柔一点，像在聊天"
      onChange={event=>setToneText(event.target.value)}/></label>
    <label className="photo-edit-field"><span>表情</span>
     <input value={expression} maxLength={40} disabled={busy||locked} placeholder="如：轻轻微笑"
      onChange={event=>setExpression(event.target.value)}/></label>
    <ul className="photo-edit-notes">
     {translateHint(prompts?.action,'动作：')&&<li>{translateHint(prompts?.action,'动作：')}</li>}
     {translateHint(prompts?.tone,'神态：')&&<li>{translateHint(prompts?.tone,'神态：')}</li>}
     {translateHint(prompts?.expression,'表情：')&&<li>{translateHint(prompts?.expression,'表情：')}</li>}
     <li className="photo-hint">提示词只影响新画出来的部分。脸和上半身保持原像素。</li>
    </ul>
   </section>
   <section className="photo-edit-group">
    <h3>按图换装</h3>
    <div className="photo-edit-garment">
     <label className={'photo-outfit-upload'+(garment?' has-file':'')}>
      <Upload size={12}/>{garment?'已选：'+garment.name:'上传服装图片'}
      <input type="file" accept="image/*" disabled={busy||locked}
       onChange={event=>{setGarment(event.target.files?.[0]||null);event.target.value=''}}/>
     </label>
     {garmentUrl&&<img src={garmentUrl} alt="已选服装"/>}
    </div>
    <label className="photo-edit-field"><span>描述</span>
     <input value={wearPrompt} maxLength={40} disabled={busy||locked} placeholder="选填，如：牛仔外套"
      onChange={event=>setWearPrompt(event.target.value)}/></label>
   </section>
   {warnings.length>0&&<ul className="photo-edit-warnings">{warnings.map(item=><li key={item}>{item}</li>)}</ul>}
   {job?.error&&<p className="photo-error-inline" role="alert">{job.error}</p>}
   {job?.result?.skipped&&<p className="photo-edit-done">这张照片已接近全身，没有生成新造型。</p>}
   {job?.status==='complete'&&job.result?.variant_id&&<p className="photo-edit-done">
    已生成「{job.result.label||'新造型'}」{job.result.metrics?.capped?'。照片里的人物偏大，这次只扩到七分身，想要全身请换一张半身照':''}{job.result.quality==='low'?'。画质可能不理想，建议换一张照片重试':''}。需要点“准备人物”后才能对话。
   </p>}
   <footer className="photo-edit-actions">
    <button type="button" className="photo-primary photo-edit-run" disabled={!!extendReason}
     title={extendReason||'把照片向下扩展成全身照'} onClick={()=>void startExtend()}>
     {running==='extend'?<Loader2 size={15} className="photo-spin"/>:<UserRound size={15}/>}生成全身照
    </button>
    <button type="button" className="photo-primary photo-edit-run" disabled={!!wearReason}
     title={wearReason||'按上传的服装图片换装'} onClick={()=>void startWear()}>
     {running==='wear'?<Loader2 size={15} className="photo-spin"/>:<Upload size={15}/>}按图换装
    </button>
   </footer>
  </div>
 </div>;
}
