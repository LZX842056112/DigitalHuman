import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Box, Camera, Check, ChevronRight, Circle, Cpu, Footprints as Walk, Hand, Loader2, Mic, Move, Play, RotateCcw, Send, ShieldCheck, Square, Trash2, Upload, Volume2, X} from 'lucide-react';
import './style.css';
import {RemoteAudioOutput,type AudioOutputState} from './native-audio-output';

type Avatar={id:string;name:string;status:string;last_job_id?:string;body_photos:string[];face_photos:string[];voice:{text:string;confirmed:boolean;reference?:string;transcription_error?:string};validation:Record<string,boolean>;render_report?:{fps:number;peak_vram_mib:number};estimated_regions:string[]};
type Job={id:string;status:string;stage:string;progress:number;error?:string;message?:string};
type Message={role:string;text:string};
const statusName:Record<string,string>={imported:'素材已导入',prepared:'素材已解析',building:'正在构建',voice_pending:'等待声音验证',ready:'可启动交互',failed:'构建未通过'};
const stageName:Record<string,string>={source_integrity:'校验原始素材',media:'解析视频和照片',prepared:'素材解析完成',body_reconstruction:'重建三维形象',head_calibration:'校准面部形象',expression_mapping:'配置口型和表情',reload_and_render:'验证人物重新加载',reconstructed:'三维人物验证完成',reference_transcription:'转写参考录音',reference_confirmation:'核对参考文字',voice_and_audio_expression:'验证声音和口型',ready:'人物包已就绪',gpu_busy:'显卡正在执行其他任务'};
async function api(path:string, init?:RequestInit){const r=await fetch('/api'+path,init);if(!r.ok){let message=await r.text();try{message=JSON.parse(message).detail}catch{}throw Error(String(message))}return r.json()}
const json=(value:unknown)=>({headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});

function App(){
 const [avatars,setAvatars]=useState<Avatar[]>([]),[selected,setSelected]=useState(''),[job,setJob]=useState<Job|null>(null);
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),[connected,setConnected]=useState(false),[status,setStatus]=useState('等待连接');
 const [messages,setMessages]=useState<Message[]>([]),[text,setText]=useState(''),[transcript,setTranscript]=useState(''),[confirmed,setConfirmed]=useState(false);
 const [action,setAction]=useState('idle'),[recording,setRecording]=useState(false),[micAllowed,setMicAllowed]=useState(false),[system,setSystem]=useState<Record<string,unknown>|null>(null);
 const [soundState,setSoundState]=useState<AudioOutputState>('off'),[soundError,setSoundError]=useState(''),[volume,setVolume]=useState(1);
 const [renderSize,setRenderSize]=useState(768),[activeSize,setActiveSize]=useState(512),[framing,setFraming]=useState('full');
 const viewCameras=useRef<Record<string,{yaw:number;pitch:number;distance:number}>>({});
 const minDistance=useRef(.3);
 const audioOutput=useRef<RemoteAudioOutput|null>(null);
 const remoteAudio=useRef<HTMLAudioElement>(null);
 const video=useRef<HTMLVideoElement>(null),peer=useRef<RTCPeerConnection|null>(null),channel=useRef<RTCDataChannel|null>(null),sid=useRef(''),local=useRef<MediaStream|null>(null);
 const camera=useRef({yaw:0,pitch:0,distance:3.2}),drag=useRef<{x:number;y:number}|null>(null),lastCamera=useRef(0),scroll=useRef<HTMLDivElement>(null);
 const current=avatars.find(a=>a.id===selected);
 const refreshVersion=useRef(0);
 async function refresh(){const version=++refreshVersion.current;const items:Avatar[]=await api('/avatars');if(version!==refreshVersion.current)return;setAvatars(items);setSelected(s=>items.some(a=>a.id===s)?s:items[0]?.id||'')}
 useEffect(()=>{refresh().catch(e=>setError(e.message));api('/system').then(setSystem).catch(()=>{});return()=>{peer.current?.close();local.current?.getTracks().forEach(t=>t.stop());audioOutput.current?.close().catch(()=>{});if(sid.current)fetch('/api/sessions/'+sid.current,{method:'DELETE',keepalive:true})}},[]);
 useEffect(()=>{const timer=setInterval(()=>refresh().catch(()=>{}),5000);return()=>clearInterval(timer)},[]);
 useEffect(()=>{setTranscript(current?.voice.text||'');setConfirmed(current?.voice.confirmed||false)},[selected,current?.voice.text,current?.voice.confirmed]);
 useEffect(()=>{let active=true;setJob(null);if(current?.last_job_id)api('/jobs/'+current.last_job_id).then(j=>{if(active)setJob(j)}).catch(e=>{if(active)setError(e.message)});return()=>{active=false}},[current?.id,current?.last_job_id]);
 useEffect(()=>{if(!job||!['queued','running'].includes(job.status))return;const timer=setInterval(()=>api('/jobs/'+job.id).then(j=>{setJob(j);if(!['running','queued'].includes(j.status))refresh().catch(()=>{})}).catch(e=>setError(e.message)),1500);return()=>clearInterval(timer)},[job?.id,job?.status]);
 useEffect(()=>{scroll.current?.scrollTo({top:scroll.current.scrollHeight,behavior:'smooth'})},[messages,status]);
 function send(value:unknown){if(channel.current?.readyState==='open')channel.current.send(JSON.stringify(value))}
 function output(){if(!audioOutput.current){audioOutput.current=new RemoteAudioOutput(()=>remoteAudio.current!,state=>{setSoundState(state);if(state==='blocked')setSoundError('声音播放被阻止。请点击播放器的播放键。');else if(state==='running')setSoundError('')});audioOutput.current.setVolume(volume)}return audioOutput.current}
 async function enableSound(){setSoundError('');try{await output().enable()}catch{setSoundError('声音未开启。请点击“开启声音”。')}}
 async function disconnect(){send({type:'interrupt'});channel.current?.close();peer.current?.close();peer.current=null;channel.current=null;local.current?.getTracks().forEach(t=>t.stop());local.current=null;await audioOutput.current?.close().catch(()=>{});audioOutput.current=null;setSoundError('');setMicAllowed(false);setRecording(false);setConnected(false);setStatus('已断开');if(video.current)video.current.srcObject=null;const id=sid.current;sid.current='';if(id)await api('/sessions/'+id,{method:'DELETE'}).catch(()=>{})}
 async function connect(){setBusy(true);setError('');void enableSound();try{
   const created=await api('/sessions',{method:'POST',...json({avatar_id:selected,render_size:renderSize})});sid.current=created.session_id;
   setActiveSize(created.render_size||512);viewCameras.current=created.views||{full:created.camera};setFraming('full');
   minDistance.current=created.camera_limits?.min_distance||.3;
   if(created.camera)camera.current={yaw:created.camera.yaw,pitch:created.camera.pitch,distance:created.camera.distance};
   const pc=new RTCPeerConnection({iceServers:[]});peer.current=pc;
   const received=new MediaStream();if(video.current)video.current.srcObject=received;
   pc.ontrack=e=>{if(peer.current!==pc)return;if(e.track.kind==='audio'){output().attach(new MediaStream([e.track]))}else{received.addTrack(e.track);video.current?.play().catch(()=>setError('画面播放被阻止，请点击画面。'))}};
   pc.onconnectionstatechange=()=>{if(['failed','closed'].includes(pc.connectionState)){setConnected(false);setStatus('连接已断开')}else if(pc.connectionState==='connected'){setConnected(true);setStatus('已连接，可以开始对话')}};
   const dc=pc.createDataChannel('control',{ordered:true});channel.current=dc;
   dc.onmessage=e=>{try{const d=JSON.parse(e.data);if(d.type==='transcript')setMessages(m=>[...m,{role:d.role,text:d.text}]);if(d.type==='status')setStatus(d.text);if(d.type==='error')setError(d.text)}catch{}};
   pc.addTransceiver('video',{direction:'recvonly'});
   try{local.current=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});local.current.getTracks().forEach(t=>pc.addTrack(t,local.current!));setMicAllowed(true)}catch{pc.addTransceiver('audio',{direction:'recvonly'});setMicAllowed(false)}
   await pc.setLocalDescription(await pc.createOffer());
   if(pc.iceGatheringState!=='complete')await new Promise<void>((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('本机 ICE 收集超时')),15000);pc.addEventListener('icegatheringstatechange',()=>{if(pc.iceGatheringState==='complete'){clearTimeout(timeout);resolve()}})});
   const answer=await api('/sessions/'+created.session_id+'/offer',{method:'POST',...json({sdp:pc.localDescription!.sdp,type:pc.localDescription!.type})});await pc.setRemoteDescription(answer);
 }catch(e){setError((e as Error).message);await disconnect()}finally{setBusy(false)}}
 async function build(stage:string){if(!current)return;setBusy(true);setError('');try{const result=await api('/avatars/'+selected+'/build',{method:'POST',...json({stage})});setJob(await api('/jobs/'+result.job_id));}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 async function deleteAvatar(){
  if(!current||working||connected||sid.current)return;
  const id=current.id;
  if(!window.confirm(`确认删除人物“${current.name}”？\n\n将删除此人物在项目内的素材副本、模型和参考声音。桌面原始文件会保留。\n删除后无法恢复。`))return;
  setBusy(true);setError('');setStatus('正在删除人物');
  try{await api('/avatars/'+id,{method:'DELETE'});++refreshVersion.current;setJob(null);setMessages([]);setText('');await refresh();setStatus('人物已删除')}
  catch(e){setError((e as Error).message);setStatus('删除未完成')}
  finally{setBusy(false)}
 }
 async function saveVoice(){setBusy(true);try{await api('/avatars/'+selected+'/voice',{method:'PUT',...json({text:transcript,confirmed,language:'zh'})});await refresh();setStatus('参考文字已保存，请执行声音验证')}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 async function importFiles(e:React.FormEvent<HTMLFormElement>){e.preventDefault();setBusy(true);setError('');try{const data=new FormData(e.currentTarget);const result=await api('/avatars/import',{method:'POST',body:data});await refresh();setSelected(result.avatar_id)}catch(e){setError((e as Error).message)}finally{setBusy(false)}}
 function orbit(force=false){if(!force&&performance.now()-lastCamera.current<70)return;lastCamera.current=performance.now();send({type:'camera',...camera.current})}
 function frameView(view:string){const preset=viewCameras.current[view];if(!preset)return;camera.current={yaw:preset.yaw,pitch:preset.pitch,distance:preset.distance};setFraming(view);send({type:'view',view})}
 function resetCamera(){frameView('full');orbit(true);send({type:'position',position:[0,0,0]})}
 function chat(e:React.FormEvent){e.preventDefault();if(!text.trim()||!connected)return;send({type:'chat_text',text:text.trim()});setText('')}
 const working=busy||!!job&&['queued','running'].includes(job.status);
 const previewMode=Array.isArray(system?.checks)&&system.checks.some((check:{name:string;ok:boolean})=>check.name==='platform'&&!check.ok);
 return <div className="studio">
  <header><div className="brand"><span className="brand-mark"><Box size={23}/></span><div><b>数字人工作室</b><small>LOCAL HUMAN STUDIO</small></div></div><div className="privacy"><ShieldCheck size={15}/> 本机推理 <span>·</span> 素材保存在 D 盘</div><span className="version">研究预览 / 01</span></header>
  {previewMode&&<div className="runtime-banner">当前页面为界面预览。WSL 环境安装完成后，才能启动三维构建和声音验证。</div>}
  <div className="workspace">
   <aside className="assets panel"><div className="panel-title"><span>01 / 人物资产</span><Circle size={8} fill="currentColor"/></div>
    <div className="asset-toolbar"><select aria-label="选择人物" value={selected} disabled={connected||busy} onChange={e=>{setSelected(e.target.value);setJob(null)}}>{!avatars.length&&<option value="">请导入素材</option>}{avatars.map(a=><option key={a.id} value={a.id}>{a.name}</option>)}</select><button type="button" className="delete-avatar" onClick={deleteAvatar} disabled={!current||working||connected||!!sid.current} title="删除所选人物"><Trash2 size={13}/>删除人物</button></div>
    {current&&<><div className="reference"><img alt="身体外观参考照片" src={'/api/avatars/'+selected+'/files/source/'+encodeURIComponent(current.body_photos[0])}/><span>身体外观参考</span></div><div className="asset-status"><span className={'dot '+(current.status==='ready'?'green':'amber')}/>{statusName[current.status]||current.status}</div><p className="hint">背面和遮挡区域使用模型估计。请对照素材检查外观。</p>
    <div className="build-actions"><button onClick={()=>build('prepare')} disabled={working||connected}>解析素材</button><button className="primary" onClick={()=>build('all')} disabled={working||connected||previewMode}>{working?<Loader2 size={15} className="spin"/>:<Box size={15}/>} 构建人物</button></div>
    {job&&<div className={'job '+(job.status==='failed'?'failed':'')}><div>{stageName[job.stage]||'处理人物任务'}<b>{Math.round(job.progress*100)}%</b></div><progress max={1} value={job.progress}/><p>{job.error||job.message||({running:'任务正在本机执行',complete:'本阶段已完成',waiting:'需要核对参考录音'} as Record<string,string>)[job.status]}</p></div>}
    <details className="voice" open={!current.voice.confirmed}><summary><Volume2 size={14}/> 声音参考 <span>{current.voice.confirmed?'已核对':'待校对'}</span></summary>{current.voice.reference?<audio controls preload="none" src={'/api/avatars/'+selected+'/files/voice/reference.wav'}/>:<p className="hint">解析视频后生成参考录音。</p>}<textarea aria-label="参考录音文本" value={transcript} onChange={e=>{setTranscript(e.target.value);setConfirmed(false)}} placeholder="试听后填写录音中的原话" rows={3}/><label className="check"><input type="checkbox" checked={confirmed} onChange={e=>setConfirmed(e.target.checked)}/>我已核对录音与文字</label><div className="build-actions"><button disabled={working||connected} onClick={saveVoice}><Check size={14}/> 保存文字</button><button disabled={working||connected||!current.voice.confirmed||previewMode} onClick={()=>build('validate')}>验证声音 <ChevronRight size={14}/></button></div></details>
    </>}
    <details><summary><Upload size={14}/> 导入新人物</summary><form className="import-form" onSubmit={importFiles}><label>人物名称<input name="name" defaultValue="新数字人" required/></label><label>人物视频<input name="video" type="file" accept="video/*" required/></label><label>身体主参考<input name="body_photos" type="file" accept="image/png,image/jpeg" required/></label><label>面部补充照片<input name="face_photos" type="file" accept="image/png,image/jpeg" multiple/></label><button type="submit" disabled={working}>导入素材</button></form></details>
   </aside>
   <main className="viewport-column"><div className="view-heading"><div><span className="eyebrow">三维交互预览</span><h1>{current?.name||'创建你的数字人'}</h1></div><span className={'connection '+(connected?'online':'')}><span className="dot"/>{connected?'实时连接':'未连接'}</span></div>
    <div className="view-controls"><label>画面<select aria-label="画面清晰度" value={renderSize} disabled={connected||busy} onChange={e=>setRenderSize(+e.target.value)}><option value={512}>512 · 标准</option><option value={768}>768 · 清晰</option><option value={1024}>1024 · 细节</option></select></label><button className={framing==='full'?'active':''} disabled={!connected} onClick={()=>frameView('full')}>全身</button><button className={framing==='portrait'?'active':''} disabled={!connected||!viewCameras.current.portrait} onClick={()=>frameView('portrait')}>面部近景</button><small>清晰模式速度较慢</small></div>
    <div className={'viewport '+(connected?'live':'')} onPointerDown={e=>{if(!connected)return;drag.current={x:e.clientX,y:e.clientY};e.currentTarget.setPointerCapture(e.pointerId)}} onPointerMove={e=>{if(!drag.current)return;camera.current.yaw+=(e.clientX-drag.current.x)*.45;camera.current.pitch=Math.max(-75,Math.min(75,camera.current.pitch+(e.clientY-drag.current.y)*.3));drag.current={x:e.clientX,y:e.clientY};orbit()}} onPointerUp={()=>{drag.current=null;orbit(true)}} onWheel={e=>{camera.current.distance=Math.max(minDistance.current,Math.min(8,camera.current.distance+e.deltaY*.003));orbit()}}>
     <div className="viewport-top"><span><Camera size={13}/> {framing==='portrait'?'面部近景':'全身视角'}</span><span>{connected?activeSize:renderSize} × {connected?activeSize:renderSize} <i/> 原生渲染</span></div>
     <video ref={video} autoPlay playsInline muted onClick={()=>{video.current?.play().catch(()=>{});void enableSound()}} aria-label="真实三维人物 WebRTC 画面"/>
     {!connected&&<div className="empty-view"><div className="wire-box"><Box size={54} strokeWidth={.8}/></div><h2>{working?'正在准备人物资产':current?.status==='ready'?'人物包已就绪':'等待真实三维人物'}</h2><p>{working?'构建日志和进度显示在左侧。':'完成重建、表情和声音检查后，在此启动交互。'}</p><button className="primary" onClick={connect} disabled={working||current?.status!=='ready'||previewMode}>{busy?<Loader2 className="spin" size={16}/>:<Play size={16}/>} 启动交互</button></div>}
     <div className="viewport-bottom"><span><Move size={13}/> 拖动旋转 · 滚轮缩放</span><button title="重置视角和位置" disabled={!connected} onClick={resetCamera}><RotateCcw size={15}/> 重置</button></div>
    </div>
    <div className="motion-bar"><div className="motion-label">人物动作</div>{[{id:'idle',label:'站立',Icon:Circle},{id:'walk',label:'步行',Icon:Walk},{id:'wave',label:'挥手',Icon:Hand}].map(({id,label,Icon})=><button key={id} disabled={!connected} className={action===id?'active':''} onClick={()=>{setAction(id);send({type:'action',action:id})}}><Icon size={17}/>{label}</button>)}<label className="position">水平位置<input aria-label="人物水平位置" type="range" min={-1} max={1} step={.05} defaultValue={0} disabled={!connected} onChange={e=>send({type:'position',position:[+e.target.value,0,0]})}/></label></div>
    <div className="audio-controls"><button onClick={()=>void enableSound()} disabled={!connected&&!busy}><Volume2 size={16}/>{soundState==='running'?'声音已开启':'开启声音'}</button><audio ref={remoteAudio} controls autoPlay aria-label="数字人实时语音" onVolumeChange={e=>{setVolume(e.currentTarget.volume);audioOutput.current?.setVolume(e.currentTarget.volume)}}/><label>音量<input aria-label="回答音量" type="range" min={0} max={1} step={.05} value={volume} onChange={e=>{const value=+e.target.value;setVolume(value);output().setVolume(value)}}/><span>{Math.round(volume*100)}%</span></label>{soundError&&<span role="alert">{soundError}</span>}</div>
    <div className="metrics"><span><Cpu size={14}/> 本地服务</span><span>{current?.render_report?`身体构建短测 ${current.render_report.fps.toFixed(1)} FPS`:'性能未测量'}</span><span>{system?'环境报告已读取':'正在读取环境'}</span></div>
   </main>
   <aside className="conversation panel"><div className="panel-title"><span>02 / 与数字人对话</span><button title="关闭会话" onClick={disconnect} disabled={!sid.current}><X size={16}/></button></div><div className="chat-info"><span className="dot green"/> 中文 · 简短回答 <span>本地模型</span></div><div className="messages" ref={scroll}>{messages.length?messages.map((m,i)=><div key={i} className={'message '+m.role}><small>{m.role==='user'?'你':'数字人'}</small><p>{m.text}</p></div>):<div className="chat-empty"><Volume2 size={28} strokeWidth={1}/><h3>从一句问候开始</h3><p>输入文字，或按住下方按钮说话。数字人会使用参考声音回答。</p></div>}</div><div className="session-status"><span className="dot"/>{status}</div><form onSubmit={chat} className="chat-form"><textarea aria-label="对话文字" value={text} onChange={e=>setText(e.target.value)} maxLength={500} placeholder="输入你想说的话…" disabled={!connected} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();chat(e)}}}/><button title="发送文字" type="submit" disabled={!connected||!text.trim()}><Send size={17}/></button></form><div className="talk-actions"><button disabled={!connected||!micAllowed} className={recording?'recording':''} onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);setRecording(true);send({type:'start_recording'})}} onPointerUp={()=>{setRecording(false);send({type:'stop_recording'})}} onPointerCancel={()=>{setRecording(false);send({type:'stop_recording'})}}><Mic size={16}/>{recording?'松开结束':'按住说话'}</button><button disabled={!connected} onClick={()=>send({type:'interrupt'})}><Square size={13}/> 打断</button></div>{connected&&!micAllowed&&<p className="hint">麦克风未授权。可继续使用文字输入。</p>}</aside>
  </div>{error&&<div role="alert" className="error-toast"><span>{error}</span><button title="关闭错误提示" onClick={()=>setError('')}><X size={17}/></button></div>}
  <footer><span>所有推理均在本机执行</span><span>真实三维重建 · 可重新加载的人物包</span></footer>
 </div>
}
createRoot(document.getElementById('root')!).render(<App/>);
