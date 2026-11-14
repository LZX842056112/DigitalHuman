import React, {useEffect, useRef, useState} from 'react';
import {Check, Loader2, Mic, Play, Send, Sparkles, Square, Trash2, Upload, Volume2, X} from 'lucide-react';
import {api,fileUrl,json} from './api';
import EditWorkbench from './edit/EditWorkbench';
import {RemoteAudioOutput,observePlayback,observeRtcSynchronization,type AudioOutputState} from './native-audio-output';
import './photo-style.css';

type Variant={id:string;label:string;photo:string;portrait?:{folder:string;width:number;height:number;fps:number}|null;validation:{photo:boolean}};
type Settings={tone:string;expression:string;tone_prompt:string};
type Avatar={id:string;name:string;status:string;photo:string;voice_video:string;voice:{confirmed:boolean;reference?:string};validation:{photo:boolean;voice:boolean};portrait?:{folder:string;width:number;height:number;fps:number};variants?:Variant[];active_variant?:string;settings?:Settings;last_job_id?:string};
type Job={id:string;status:string;stage:string;progress:number;error?:string;message?:string};
type Message={role:string;text:string;partial?:boolean};
type RtcProbe={state:string;samples?:number;maxAbsOffsetMs?:number;p95AbsOffsetMs?:number;video?:{framesDecoded?:number;framesDropped?:number;framesPerSecond?:number}};
type TurnTiming={generation:number;input_kind:'text'|'audio';status:string;asr_ms:number|null;llm_ms:number|null;tts_first_ms:number|null;portrait_first_ms:number|null;first_enqueue_ms:number|null;total_ms:number|null;tts_first_chunk_ms?:number|null;portrait_first_chunk_ms?:number|null;first_pair_ready_ms?:number|null;playback_start_ms?:number|null;stream_retries?:number;llm_first_ms?:number|null;llm_first_sentence_ms?:number|null};
type RuntimeProbe={delivery_mode?:string;stream_underruns?:number;paired_buffer_seconds?:number;stream_fallback_reason?:string|null};
const milliseconds=(value:number|null|undefined)=>typeof value==='number'?value.toFixed(1)+' 毫秒':'未测量';
const timingStatusName:Record<string,string>={running:'正在处理',completed:'已完成',interrupted:'已打断',failed:'未完成'};
const deliveryName:Record<string,string>={buffered:'整句准备',streaming_serial:'串行分块',streaming_parallel:'并行分块'};
const toneOptions:[string,string][]=[['natural','自然'],['warm','亲切'],['professional','专业'],['lively','活泼']];
const expressionOptions:[string,string][]=[['natural','自然'],['smile','微笑'],['serious','严肃'],['surprise','惊讶']];
const activeJob=(job:Job|null)=>!!job&&['queued','running'].includes(job.status);
const statusName:Record<string,string>={imported:'素材已导入',preparing:'正在准备',voice_pending:'请试听参考声音',ready:'可以开始对话',failed:'准备未完成'};
const stageName:Record<string,string>={queued:'等待处理',import:'导入素材',audio:'提取参考声音',photo:'准备人物画面',voice:'检查克隆声音',complete:'准备完成',source_integrity:'检查素材',media:'读取照片和声音',reference_confirmation:'请试听参考声音',voice_reference:'准备参考声音',portrait:'准备人物画面',voice_and_audio_expression:'检查声音和口型',ready:'准备完成',gpu_busy:'等待本机任务完成',analyze:'分析人物',canvas:'准备画布',diffuse:'生成画面',compose:'拼接接缝',verify:'检查人脸'};
const avatarUrl=(id:string)=>'/avatars/'+encodeURIComponent(id);

export default function PhotoApp(){
 const [avatars,setAvatars]=useState<Avatar[]>([]),[selected,setSelected]=useState(''),[job,setJob]=useState<Job|null>(null);
 const [busy,setBusy]=useState(false),[connecting,setConnecting]=useState(false),[connected,setConnected]=useState(false);
 const [error,setError]=useState(''),[status,setStatus]=useState('等待连接'),[messages,setMessages]=useState<Message[]>([]),[text,setText]=useState('');
 const [recording,setRecording]=useState(false),[micAllowed,setMicAllowed]=useState(false),[generating,setGenerating]=useState(false);
 const [volume,setVolume]=useState(1),[soundState,setSoundState]=useState<AudioOutputState>('off'),[soundError,setSoundError]=useState('');
 const [rtcProbe,setRtcProbe]=useState<RtcProbe|null>(null),[turnTiming,setTurnTiming]=useState<TurnTiming|null>(null);
 const [runtimeProbe,setRuntimeProbe]=useState<RuntimeProbe|null>(null);
 const [settings,setSettings]=useState<Settings>({tone:'natural',expression:'natural',tone_prompt:''});
 const [editing,setEditing]=useState(false),[editBusy,setEditBusy]=useState(false);
 const [outfitOptions,setOutfitOptions]=useState<{wear:boolean}>({wear:false});
 const [outfitPrompt,setOutfitPrompt]=useState(''),[outfitFile,setOutfitFile]=useState<File|null>(null);
 const mounted=useRef(true),operation=useRef(false),sessionInFlight=useRef(false),isRecording=useRef(false),refreshVersion=useRef(0),jobVersion=useRef(0),sessionVersion=useRef(0);
 const peer=useRef<RTCPeerConnection|null>(null),channel=useRef<RTCDataChannel|null>(null),sid=useRef(''),local=useRef<MediaStream|null>(null);
 const audioOutput=useRef<RemoteAudioOutput|null>(null),remoteAudio=useRef<HTMLAudioElement>(null),referenceAudio=useRef<HTMLAudioElement>(null),video=useRef<HTMLVideoElement>(null),scroll=useRef<HTMLDivElement>(null);
 const connectionTimer=useRef<ReturnType<typeof setTimeout>|null>(null);
 const stopSynchronization=useRef<(()=>void)|null>(null);
 const current=avatars.find(a=>a.id===selected);
 const working=busy||activeJob(job)||current?.status==='preparing',sessionActive=connecting||connected;
 const locked=working||sessionActive||editBusy;
 const editOpen=editing&&!sessionActive&&!!current;
 const diagnostics=new URLSearchParams(location.search).get('diagnostics')==='1';
 useEffect(()=>{
  if(new URLSearchParams(location.search).get('diagnostics')==='1'&&video.current&&remoteAudio.current)
   return observePlayback(video.current,remoteAudio.current);
 },[]);
 useEffect(()=>{
  if(!diagnostics)return;
  const timer=setInterval(()=>{try{const raw=video.current?.dataset.rtcSynchronization;if(raw)setRtcProbe(JSON.parse(raw))}catch{}},1000);
  return ()=>clearInterval(timer);
 },[diagnostics]);
 useEffect(()=>{
  if(!diagnostics||!connected||!sid.current)return;
  const id=sid.current,version=sessionVersion.current;let stopped=false,pending=false;
  async function sample(){
   if(pending)return;pending=true;
   try{const result=await api('/sessions/'+encodeURIComponent(id)+'/metrics');
    if(!stopped&&sessionVersion.current===version&&sid.current===id){setTurnTiming(result.last_turn_timing||null);setRuntimeProbe(result)}
   }catch{}finally{pending=false}
  }
  void sample();const timer=setInterval(()=>void sample(),1000);
  return ()=>{stopped=true;clearInterval(timer)};
 },[diagnostics,connected]);
 async function refresh(preferred?:string){
  const version=++refreshVersion.current;const items:Avatar[]=await api('/avatars');
  if(!mounted.current||version!==refreshVersion.current)return;
  setAvatars(items);setSelected(previous=>items.some(a=>a.id===(preferred||previous))?(preferred||previous):items[0]?.id||'');
 }
 function send(value:unknown){if(channel.current?.readyState!=='open')return false;channel.current.send(JSON.stringify(value));return true}
 function output(){
  if(!audioOutput.current){
   audioOutput.current=new RemoteAudioOutput(()=>remoteAudio.current!,state=>{if(!mounted.current)return;setSoundState(state);if(state==='blocked')setSoundError('请点击播放器的播放键。');else if(state==='running')setSoundError('')});
   audioOutput.current.setVolume(volume);
  }
  return audioOutput.current;
 }
 async function enableSound(){setSoundError('');try{await output().enable()}catch{if(mounted.current)setSoundError('请点击“开启声音”重试。')}}
 function releaseSession(){
  ++sessionVersion.current;
  sessionInFlight.current=false;isRecording.current=false;
  if(connectionTimer.current)clearTimeout(connectionTimer.current);connectionTimer.current=null;
  stopSynchronization.current?.();stopSynchronization.current=null;
  const pc=peer.current;peer.current=null;pc?.close();channel.current?.close();channel.current=null;
  local.current?.getTracks().forEach(track=>track.stop());local.current=null;
  void audioOutput.current?.close().catch(()=>{});audioOutput.current=null;
  if(video.current)video.current.srcObject=null;
  if(video.current)delete video.current.dataset.microphone;
  const id=sid.current;sid.current='';
  if(id)void fetch('/api/sessions/'+encodeURIComponent(id),{method:'DELETE',keepalive:true}).catch(()=>{});
 }
 function disconnect(){
  send({type:'interrupt'});releaseSession();setConnected(false);setConnecting(false);setRecording(false);setMicAllowed(false);setGenerating(false);setSoundError('');setStatus('已断开');
 }
 useEffect(()=>{
  void api('/outfit/options').then(value=>{if(mounted.current)setOutfitOptions(value)}).catch(()=>{});
 },[]);
 useEffect(()=>{
  mounted.current=true;void refresh().catch(e=>{if(mounted.current)setError(e.message)});
  const timer=setInterval(()=>void refresh().catch(()=>{}),5000);
  return()=>{mounted.current=false;++refreshVersion.current;++jobVersion.current;clearInterval(timer);releaseSession()};
 },[]);
 useEffect(()=>{
  const version=++jobVersion.current;setJob(previous=>previous?.id===current?.last_job_id?previous:null);
  if(current?.last_job_id)void api('/jobs/'+encodeURIComponent(current.last_job_id)).then(result=>{if(mounted.current&&version===jobVersion.current)setJob(result)}).catch(e=>{if(mounted.current&&version===jobVersion.current)setError(e.message)});
  return()=>{++jobVersion.current};
 },[selected,current?.last_job_id]);
 useEffect(()=>{
  if(!job||!activeJob(job))return;
  let active=true,pending=false;
  const timer=setInterval(async()=>{
   if(pending)return;pending=true;
   try{const result:Job=await api('/jobs/'+encodeURIComponent(job.id));if(!active)return;setJob(result);if(!activeJob(result))await refresh()}
   catch(e){if(active)setError((e as Error).message)}finally{pending=false}
  },1500);
  return()=>{active=false;clearInterval(timer)};
 },[job?.id,job?.status]);
 useEffect(()=>{scroll.current?.scrollTo({top:scroll.current.scrollHeight,behavior:'smooth'})},[messages,status]);
 useEffect(()=>{const value=current?.settings;if(value)setSettings({tone:value.tone,expression:value.expression,tone_prompt:value.tone_prompt||''})},[current?.id,current?.settings?.tone,current?.settings?.expression,current?.settings?.tone_prompt]);
 async function mutate(action:()=>Promise<void>){
  if(operation.current||locked||sid.current)return;
  operation.current=true;setBusy(true);setError('');
  try{await action()}catch(e){if(mounted.current)setError((e as Error).message)}finally{operation.current=false;if(mounted.current)setBusy(false)}
 }
 async function startBuild(id:string){
  ++jobVersion.current;
  const result=await api(avatarUrl(id)+'/build',{method:'POST'});
  if(mounted.current)setJob({id:result.job_id,status:'queued',stage:'',progress:0});
  try{const next:Job=await api('/jobs/'+encodeURIComponent(result.job_id));if(mounted.current)setJob(next)}finally{await refresh(id)}
 }
 function build(){if(current)void mutate(()=>startBuild(current.id))}
 function confirmVoice(){if(current)void mutate(async()=>{await api(avatarUrl(current.id)+'/voice',{method:'PUT',...json({confirmed:true})});await refresh(current.id);await startBuild(current.id)})}
 function importFiles(event:React.FormEvent<HTMLFormElement>){
  event.preventDefault();const form=event.currentTarget,data=new FormData(form);
  void mutate(async()=>{const result=await api('/avatars/import',{method:'POST',body:data});await refresh(result.avatar_id||result.id);form.reset();setMessages([]);setStatus('素材已导入。请点击“准备人物”。')});
 }
 function replaceFile(kind:'photo'|'voice-video',file:File|undefined){
  if(!current||!file)return;const id=current.id,data=new FormData();data.set(kind==='photo'?'photo':'voice_video',file);
  void mutate(async()=>{await api(avatarUrl(id)+'/'+kind,{method:'PUT',body:data});setJob(null);await refresh(id);setStatus('素材已更新。请重新准备人物。')});
 }
 function applySettings(patch:Partial<Settings>){
  if(!current||working)return;
  const previous=settings,next={...settings,...patch};setSettings(next);
  if(connected&&send({type:'settings',...patch}))return;
  void api(avatarUrl(current.id)+'/settings',{method:'PUT',...json(patch)})
   .then(()=>refresh(current.id))
   .catch(error=>{setSettings(previous);setError((error as Error).message)});
 }
 function activateVariant(variantId:string){
  if(!current||locked)return;const id=current.id;
  void mutate(async()=>{await api(avatarUrl(id)+'/variants/'+variantId+'/active',{method:'PUT'});await refresh(id);setStatus('已切换造型，下次连接生效。')});
 }
 function addVariant(file:File|undefined){
  if(!current||!file)return;const id=current.id,data=new FormData();data.set('photo',file);
  void mutate(async()=>{await api(avatarUrl(id)+'/variants',{method:'POST',body:data});await refresh(id);setStatus('已添加造型。请点击“准备人物”。')});
 }
 function chooseGarment(file:File|undefined){
  setOutfitFile(file||null);
 }
 async function applyGarment(){
  if(!current||!outfitFile||locked)return;const id=current.id;
  const data=new FormData();data.set('garment',outfitFile);
  if(outfitPrompt.trim())data.set('prompt',outfitPrompt.trim());
  setStatus('正在按上传的衣服换装，约需一分钟…');
  void mutate(async()=>{
   const person=await api(avatarUrl(id)+'/outfit/wear',{method:'POST',body:data});
   setOutfitFile(null);
   await refresh(id);
   const label=(person.variants||[]).slice(-1)[0]?.label||'换装';
   setStatus(`已生成「${label}」。请点击“准备人物”后切换使用。`);
  });
 }
 function replaceVariant(variantId:string,file:File|undefined){
  if(!current||!file)return;const id=current.id,data=new FormData();data.set('photo',file);
  void mutate(async()=>{await api(avatarUrl(id)+'/variants/'+variantId,{method:'PUT',body:data});setJob(null);await refresh(id);setStatus('造型照片已更新。请重新准备人物。')});
 }
 function deleteVariant(variantId:string){
  if(!current||locked)return;const id=current.id;
  if(!window.confirm('确认删除这套造型？照片与它的画面缓存会从人物目录移除。'))return;
  void mutate(async()=>{await api(avatarUrl(id)+'/variants/'+variantId,{method:'DELETE'});await refresh(id)});
 }
 function deleteAvatar(){
  if(!current||locked)return;
  const id=current.id;
  if(!window.confirm(`确认删除人物“${current.name}”？\n\n此操作只删除项目内的照片、视频、参考声音和生成结果副本。原始文件会保留。\n删除后无法恢复。`))return;
  void mutate(async()=>{await api(avatarUrl(id),{method:'DELETE'});++jobVersion.current;setJob(null);setMessages([]);setText('');await refresh();setStatus('人物已删除')});
 }
 async function connect(){
  if(operation.current||sessionInFlight.current||locked||!current||current.status!=='ready'||sid.current)return;
  sessionInFlight.current=true;referenceAudio.current?.pause();
   const version=++sessionVersion.current;setConnecting(true);setError('');setStatus('正在连接');setMessages([]);setTurnTiming(null);setRtcProbe(null);setRuntimeProbe(null);void enableSound();
  const valid=()=>mounted.current&&version===sessionVersion.current;
  try{
   const created=await api('/sessions',{method:'POST',...json({avatar_id:current.id})});
   if(!valid()){void fetch('/api/sessions/'+encodeURIComponent(created.session_id),{method:'DELETE',keepalive:true}).catch(()=>{});return}
   sid.current=created.session_id;
   const pc=new RTCPeerConnection({iceServers:[]});peer.current=pc;
   const received=new MediaStream();if(video.current)video.current.srcObject=received;
   if(new URLSearchParams(location.search).get('diagnostics')==='1'&&video.current&&remoteAudio.current)
    stopSynchronization.current=observeRtcSynchronization(pc,received,video.current,remoteAudio.current);
   connectionTimer.current=setTimeout(()=>{if(valid()){setError('连接超时。请重新连接。');disconnect()}},30000);
   pc.ontrack=event=>{if(!valid())return;received.addTrack(event.track);if(event.track.kind==='audio')output().attach(received);else{void video.current?.play().catch(()=>{if(valid())setError('请点击画面开始播放。')})}};
   pc.onconnectionstatechange=()=>{
    if(!valid())return;
    if(['failed','closed','disconnected'].includes(pc.connectionState)){setError('连接已断开。请重新连接。');disconnect()}
   };
   const dc=pc.createDataChannel('control',{ordered:true});channel.current=dc;
   dc.onopen=()=>{if(!valid())return;if(connectionTimer.current)clearTimeout(connectionTimer.current);connectionTimer.current=null;setConnecting(false);setConnected(true);setStatus('已连接。可以开始对话。')};
   dc.onclose=()=>{if(valid()){setError('对话连接已关闭。请重新连接。');disconnect()}};
   dc.onmessage=event=>{
    if(!valid())return;
    try{
     const data=JSON.parse(event.data);
     if(data.type==='reply_partial'&&typeof data.text==='string')setMessages(previous=>{
      const last=previous[previous.length-1];
      const entry={role:'assistant',text:data.text,partial:true};
      return last?.role==='assistant'&&last.partial?[...previous.slice(0,-1),entry]:[...previous.slice(-79),entry];
     });
     if(data.type==='transcript'&&typeof data.text==='string')setMessages(previous=>{
      const last=previous[previous.length-1];
      const entry={role:data.role==='user'?'user':'assistant',text:data.text};
      if(last?.role==='assistant'&&last.partial)return [...previous.slice(0,-1),entry];
      return [...previous.slice(-79),entry];
     });
     if(data.type==='status'&&typeof data.text==='string')setStatus(data.text);
     if(data.type==='error'){setError(data.text||'对话未完成。请重试。');setGenerating(false)}
     if(data.type==='turn_complete'){setGenerating(false);setStatus('回答完成')}
     if(diagnostics&&['turn_complete','error'].includes(data.type))
      void api('/sessions/'+encodeURIComponent(created.session_id)+'/metrics').then(result=>{if(valid())setTurnTiming(result.last_turn_timing||null)}).catch(()=>{});
    }catch{}
   };
   pc.addTransceiver('video',{direction:'recvonly'});
   try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
    if(!valid()){stream.getTracks().forEach(track=>track.stop());return}
    local.current=stream;stream.getTracks().forEach(track=>pc.addTrack(track,stream));setMicAllowed(true);
    if(new URLSearchParams(location.search).get('diagnostics')==='1'&&video.current){
     const microphone=stream.getAudioTracks()[0],settings=microphone?.getSettings();
     video.current.dataset.microphone=JSON.stringify({label:microphone?.label,
      sampleRate:settings?.sampleRate,channelCount:settings?.channelCount,
      echoCancellation:settings?.echoCancellation,noiseSuppression:settings?.noiseSuppression});
    }
   }catch{if(!valid())return;pc.addTransceiver('audio',{direction:'recvonly'});setMicAllowed(false)}
   await pc.setLocalDescription(await pc.createOffer());
   if(pc.iceGatheringState!=='complete')await new Promise<void>((resolve,reject)=>{
    const finish=()=>{if(pc.iceGatheringState==='complete'){clearTimeout(timer);pc.removeEventListener('icegatheringstatechange',finish);resolve()}};
    const timer=setTimeout(()=>{pc.removeEventListener('icegatheringstatechange',finish);reject(Error('连接准备超时。请重试。'))},15000);
    pc.addEventListener('icegatheringstatechange',finish);finish();
   });
   if(!valid())return;
   const answer=await api('/sessions/'+encodeURIComponent(created.session_id)+'/offer',{method:'POST',...json({sdp:pc.localDescription!.sdp,type:pc.localDescription!.type})});
   if(valid())await pc.setRemoteDescription(answer);
  }catch(e){if(valid()){setError((e as Error).message);disconnect()}}
 }
 function chat(event:React.FormEvent){event.preventDefault();if(!connected||generating||!text.trim())return;if(send({type:'chat_text',text:text.trim()})){setText('');setGenerating(true);setStatus('正在准备回答')}}
 function startRecording(){if(!connected||!micAllowed||isRecording.current)return;send({type:'interrupt'});if(send({type:'start_recording'})){isRecording.current=true;setRecording(true);setGenerating(false);setStatus('正在录音')}}
 function stopRecording(){if(!isRecording.current)return;isRecording.current=false;setRecording(false);if(send({type:'stop_recording'})){setGenerating(true);setStatus('正在识别并准备回答')}}
 function interrupt(){send({type:'interrupt'});isRecording.current=false;setRecording(false);setGenerating(false);setStatus('已打断。可以继续对话。')}
 return <div className="photo-studio">
  <header><div className="photo-brand"><span><Volume2 size={22}/></span><div><b>照片数字人</b><small>本机照片对话</small></div></div><p>一张照片，一段声音，开始对话。</p></header>
  <div className="photo-workspace">
   <aside className="photo-panel photo-assets"><h2>人物素材</h2><div className="photo-toolbar"><select aria-label="选择人物" value={selected} disabled={locked} onChange={event=>{setSelected(event.target.value);setMessages([]);setText('');setStatus('等待连接')}}>{!avatars.length&&<option value="">请导入人物</option>}{avatars.map(avatar=><option key={avatar.id} value={avatar.id}>{avatar.name}</option>)}</select><button aria-label="删除人物" title="删除人物" disabled={!current||locked} onClick={deleteAvatar}><Trash2 size={15}/></button></div>
    {current&&<><div className="photo-thumbnail"><img src={fileUrl(current.id,current.photo)} alt="当前人物照片"/></div><p className="photo-asset-status">{statusName[current.status]||'等待准备'}</p>
    <section className="photo-variants" aria-label="造型"><div className="photo-variant-list">
     {(current.variants||[]).map(variant=><div key={variant.id} className={'photo-variant'+(variant.id===current.active_variant?' active':'')}><button type="button" title={variant.portrait?'切换到这个造型':'该造型尚未准备'} disabled={locked||!variant.portrait} onClick={()=>activateVariant(variant.id)}><img src={fileUrl(current.id,variant.photo)} alt={variant.label}/></button><div className="photo-variant-tools"><label title="替换照片"><Upload size={11}/><input type="file" accept="image/*" disabled={locked} onChange={event=>{replaceVariant(variant.id,event.target.files?.[0]);event.target.value=''}}/></label><button type="button" title="删除造型" disabled={locked||(current.variants||[]).length<2} onClick={()=>deleteVariant(variant.id)}><Trash2 size={11}/></button></div><small>{variant.label}</small></div>)}
     <label className="photo-variant-add" title="添加造型"><Upload size={14}/><input type="file" accept="image/*" disabled={locked} onChange={event=>{addVariant(event.target.files?.[0]);event.target.value=''}}/><span>添加</span></label>
    </div><p className="photo-hint">造型=一张照片。可以换装生成，也可以上传另一张照片；切换后下次连接生效。</p></section>
    <section className="photo-outfit" aria-label="换装"><div className="photo-outfit-row"><span>描述</span><input className="photo-outfit-prompt" aria-label="衣服描述" placeholder="选填，如：牛仔外套" value={outfitPrompt} maxLength={40} disabled={locked||!outfitOptions.wear} onChange={event=>setOutfitPrompt(event.target.value)}/></div><div className="photo-outfit-row"><span>服装</span><label className={'photo-outfit-upload'+(outfitFile?' has-file':'')} title="上传一张服装图片，按图给人物换装"><Upload size={12}/>{outfitOptions.wear?(outfitFile?'已选：'+outfitFile.name:'上传服装图片'):'未装换装模型'}<input type="file" accept="image/*" disabled={locked||!outfitOptions.wear} onChange={event=>{chooseGarment(event.target.files?.[0]);event.target.value=''}}/></label></div><button type="button" className="photo-outfit-apply" disabled={locked||!outfitOptions.wear||!outfitFile} onClick={()=>void applyGarment()}>开始换装</button><p className="photo-hint">上传你自己的服装图片，按图给照片里的人物换装；填一句描述更准。脸和背景不动，结果作为一套新造型，准备后切用。</p></section>
    <button type="button" className="photo-wide photo-edit-open" disabled={locked||!current} title={locked?'请先停止准备任务或结束对话':'打开照片编辑'} onClick={()=>setEditing(true)}><Sparkles size={14}/>编辑照片</button>
    <section className="photo-style" aria-label="语气与表情">
     <div className="photo-style-row"><span>语气</span><div className="photo-segments">{toneOptions.map(([value,label])=><button key={value} type="button" className={settings.tone===value?'on':''} disabled={working} onClick={()=>applySettings({tone:value})}>{label}</button>)}</div></div>
     <div className="photo-style-row"><span>自定义</span><input className="photo-outfit-prompt" aria-label="自定义语气" placeholder="选填，如：像跟朋友闲聊" value={settings.tone_prompt} maxLength={60} disabled={working} onChange={event=>setSettings({...settings,tone_prompt:event.target.value})} onBlur={event=>applySettings({tone_prompt:event.target.value})} onKeyDown={event=>{if(event.key==='Enter'){event.preventDefault();applySettings({tone_prompt:event.currentTarget.value})}}}/></div>
     <div className="photo-style-row"><span>表情</span><div className="photo-segments">{expressionOptions.map(([value,label])=><button key={value} type="button" className={settings.expression===value?'on':''} disabled={working} onClick={()=>applySettings({expression:value})}>{label}</button>)}</div></div>
     <p className="photo-hint">语气改措辞与平语发音，自定义一句会同时写进画面神态；表情只在平语回答时生效，有情绪的回答按情绪演。连接中改动立即生效。</p>
    </section>
    <fieldset disabled={locked}><label>更换照片<input type="file" accept="image/*" onChange={event=>{replaceFile('photo',event.target.files?.[0]);event.target.value=''}}/></label><label>声音来源视频<input type="file" accept="video/*" onChange={event=>{replaceFile('voice-video',event.target.files?.[0]);event.target.value=''}}/></label></fieldset><p className="photo-hint">{current.voice_video.split('/').pop()}<br/>视频仅用于提取声音。需至少六秒清晰语音，音轨最长十分钟。</p><button className="photo-primary photo-wide" disabled={locked} onClick={build}>{working?<Loader2 size={15} className="photo-spin"/>:<Play size={15}/>}准备人物</button>
    {job&&<div className="photo-job" role="status"><div>{stageName[job.stage]||'准备人物'}<span>{Math.round(Math.max(0,Math.min(1,job.progress||0))*100)}%</span></div><progress max={1} value={Math.max(0,Math.min(1,job.progress||0))}/><p>{job.error||job.message||(activeJob(job)?'正在本机处理。':job.status==='waiting'?(job.stage==='reference_confirmation'?'请试听参考声音，确认后继续准备。':'等待你完成当前操作。'):job.status==='complete'?'准备已完成。':job.status==='failed'?'准备未完成，请重试。':'等待下一步操作。')}</p></div>}
    <section className="photo-voice"><h3>参考声音 · {current.voice.confirmed?'已确认':'待确认'}</h3>{current.voice.reference?<><audio ref={referenceAudio} controls={!sessionActive} preload="none" src={fileUrl(current.id,current.voice.reference)} aria-label="参考声音试听" onPlay={event=>{if(sessionInFlight.current)event.currentTarget.pause()}}/><button className="photo-wide" disabled={locked||current.voice.confirmed} onClick={confirmVoice}><Check size={14}/>{current.voice.confirmed?'已确认声音':'确认声音并继续准备'}</button></>:<p className="photo-hint">准备人物后，在此试听参考声音。</p>}</section></>}
    <details open={!current}><summary><Upload size={14}/>导入新人物</summary><form onSubmit={importFiles}><fieldset disabled={locked}><label>人物名称<input name="name" defaultValue="新数字人" maxLength={80} required/></label><label>人物照片<input name="photo" type="file" accept="image/*" required/></label><label>声音来源视频<input name="voice_video" type="file" accept="video/*" required/></label><button className="photo-wide" type="submit" disabled={locked}>导入素材</button></fieldset></form></details>
   </aside>
   <main className="photo-main"><div className="photo-view-heading"><div><small>照片对话</small><h1>{current?.name||'创建你的数字人'}</h1></div><span>{connected?'已连接':connecting?'正在连接':editOpen?'编辑中':'未连接'}</span></div>{editOpen&&current&&<EditWorkbench avatarId={current.id} variantId={current.active_variant||(current.variants||[])[0]?.id||''} variants={current.variants||[]} locked={working||sessionActive} onClose={()=>setEditing(false)} onBusy={setEditBusy} onDone={()=>{void refresh(current.id);setStatus('已生成新造型，准备人物后即可对话。')}} onError={setError}/>}{!editOpen&&<><div className="photo-view"><video ref={video} autoPlay playsInline muted hidden={!connected} aria-label="人物实时画面" onClick={()=>{void video.current?.play().catch(()=>{});void enableSound()}}/>{!connected&&(current?<img src={fileUrl(current.id,current.photo)} alt="人物原照片"/>:<div className="photo-placeholder">请先导入照片和声音来源视频。</div>)}</div><div className="photo-session-actions"><button className="photo-primary" onClick={()=>void connect()} disabled={locked||editing||current?.status!=='ready'}>{connecting?<Loader2 size={15} className="photo-spin"/>:<Play size={15}/>}开始对话</button><button disabled={!sessionActive} onClick={disconnect}><X size={15}/>断开连接</button></div><div className="photo-audio"><button onClick={()=>void enableSound()} disabled={!sessionActive}><Volume2 size={15}/>{soundState==='running'?'声音已开启':'开启声音'}</button><audio ref={remoteAudio} controls autoPlay aria-label="数字人实时语音" onVolumeChange={event=>{setVolume(event.currentTarget.volume);audioOutput.current?.setVolume(event.currentTarget.volume)}}/><label>音量<input aria-label="回答音量" type="range" min={0} max={1} step={.05} value={volume} onChange={event=>{const value=+event.target.value;setVolume(value);output().setVolume(value)}}/><span>{Math.round(volume*100)}%</span></label>{soundError&&<p role="alert">{soundError}</p>}</div><p className="photo-hint">照片按原比例显示。人物按所选语气与表情呈现；有情绪的回答优先按情绪演。</p></>}</main>
   <aside className="photo-panel photo-conversation"><h2>与数字人对话</h2><p className="photo-hint">输入文字，或按住麦克风按钮说话。</p><div className="photo-messages" ref={scroll}>{messages.length?messages.map((message,index)=><div key={index} className={'photo-message '+message.role}><small>{message.role==='user'?'你':'数字人'}</small><p>{message.text}</p></div>):<div className="photo-chat-empty"><Volume2 size={30}/><p>从一句问候开始。</p></div>}</div><p className="photo-status" role="status">{generating&&<Loader2 size={13} className="photo-spin"/>}{status}</p><form className="photo-chat-form" onSubmit={chat}><textarea aria-label="对话文字" placeholder="输入你想说的话…" value={text} maxLength={500} disabled={!connected||recording} onChange={event=>setText(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.nativeEvent.isComposing){event.preventDefault();chat(event)}}}/><button aria-label="发送文字" type="submit" disabled={!connected||generating||recording||!text.trim()}><Send size={17}/></button></form><div className="photo-talk"><button disabled={!connected||!micAllowed} className={recording?'photo-recording':''} onPointerDown={event=>{if(event.button!==0)return;event.currentTarget.setPointerCapture(event.pointerId);startRecording()}} onPointerUp={stopRecording} onPointerCancel={stopRecording} onLostPointerCapture={stopRecording} onKeyDown={event=>{if((event.key===' '||event.key==='Enter')&&!event.repeat){event.preventDefault();startRecording()}}} onKeyUp={event=>{if(event.key===' '||event.key==='Enter'){event.preventDefault();stopRecording()}}} onBlur={stopRecording}><Mic size={15}/>{recording?'松开结束':'按住说话'}</button><button disabled={!connected} onClick={interrupt}><Square size={13}/>打断</button></div>{connected&&!micAllowed&&<p className="photo-hint">麦克风未授权。可以继续输入文字。</p>}</aside>
  </div>
  {diagnostics&&connected&&<section className="photo-panel" aria-label="运行诊断"><h2>运行诊断</h2>
   <p>生成方式：{deliveryName[runtimeProbe?.delivery_mode||'']||'未测量'}；缓冲停顿：{runtimeProbe?.stream_underruns??'未测量'} 次；待播音画：{runtimeProbe?.paired_buffer_seconds?.toFixed(2)??'未测量'} 秒</p>
   {runtimeProbe?.stream_fallback_reason&&<p>{runtimeProbe.stream_fallback_reason}</p>}
   {rtcProbe?.state==='measured'&&!!rtcProbe.samples?<>
    <p>同步采样：{rtcProbe.samples} 次；最大音画差：{milliseconds(rtcProbe.maxAbsOffsetMs)}；P95：{milliseconds(rtcProbe.p95AbsOffsetMs)}</p>
    <p>解码帧：{rtcProbe.video?.framesDecoded??'未测量'}；接收端丢帧：{rtcProbe.video?.framesDropped??'未测量'}；瞬时帧率：{rtcProbe.video?.framesPerSecond??'未测量'} FPS</p>
   </>:<p>音画同步：未测量。请保持原生音频正在播放，并使用支持播放时间统计的浏览器。</p>}
   {turnTiming&&<><h3>最近一轮 · {turnTiming.generation} · {timingStatusName[turnTiming.status]||'状态未知'}</h3>
    <p>识别：{turnTiming.input_kind==='text'?'不适用':milliseconds(turnTiming.asr_ms)}；回答生成：{milliseconds(turnTiming.llm_ms)}；首句声音：{milliseconds(turnTiming.tts_first_ms)}；首句画面：{milliseconds(turnTiming.portrait_first_ms)}</p>
    <p>回答首字：{milliseconds(turnTiming.llm_first_ms)}；首句文字：{milliseconds(turnTiming.llm_first_sentence_ms)}</p>
    <p>首段就绪：{milliseconds(turnTiming.first_enqueue_ms)}；整轮：{milliseconds(turnTiming.total_ms)}</p>
    {runtimeProbe?.delivery_mode?.startsWith('streaming')&&<><p>首个声音块：{milliseconds(turnTiming.tts_first_chunk_ms)}；首个画面块：{milliseconds(turnTiming.portrait_first_chunk_ms)}</p><p>首对音画：{milliseconds(turnTiming.first_pair_ready_ms)}；播放队列开始输出：{milliseconds(turnTiming.playback_start_ms)}；本轮重试：{turnTiming.stream_retries??0} 次</p></>}
   </>}
   <p className="photo-hint">首句就绪表示已进入播放队列。瞬时帧率不替代连续运行验收。</p>
  </section>}
  <footer>素材和推理保留在本机。</footer>{error&&<div className="photo-error" role="alert"><span>{error}</span><button aria-label="关闭错误提示" onClick={()=>setError('')}><X size={16}/></button></div>}
 </div>;
}
