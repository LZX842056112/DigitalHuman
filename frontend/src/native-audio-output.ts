export type AudioOutputState='off'|'running'|'blocked';

/** Keep remote speech on the browser's native MediaStream playback path. */
export class RemoteAudioOutput {
 private element:HTMLAudioElement|null=null;
 private createElement:()=>HTMLAudioElement;
 private notify:(state:AudioOutputState)=>void;
 private volume=1;
 private closed=false;
 private attempt=0;

 constructor(createElement:()=>HTMLAudioElement,notify:(state:AudioOutputState)=>void){
  this.createElement=createElement;this.notify=notify;
 }

 private player(){
  if(!this.element){
   const element=this.createElement();this.element=element;
   element.onplaying=()=>this.updateState();element.onpause=()=>this.updateState();
   element.onvolumechange=()=>this.updateState();
  }
  return this.element;
 }

 private updateState(){
  if(!this.closed&&this.element)this.notify(this.element.paused||this.element.muted||this.element.volume===0?'off':'running');
 }

 async enable(){
  if(this.closed)throw Error('声音输出已关闭');
  const element=this.player();element.muted=false;element.volume=this.volume;
  if(!element.srcObject)return;
  const attempt=++this.attempt;
  try{
   // Invoke play inside the click, before any asynchronous work.
   await element.play();if(attempt===this.attempt)this.updateState();
  }catch(error){if(!this.closed&&attempt===this.attempt)this.notify('blocked');throw error}
 }

 attach(stream:MediaStream){
  if(this.closed)return;
  this.player().srcObject=stream;void this.enable().catch(()=>{});
 }

 setVolume(value:number){
  if(!Number.isFinite(value))return;
  this.volume=Math.max(0,Math.min(1,value));if(this.element&&this.element.volume!==this.volume)this.element.volume=this.volume;
 }

 async close(){
  if(this.closed)return;this.closed=true;++this.attempt;
  if(this.element){
   this.element.pause();this.element.srcObject=null;
   this.element.onplaying=null;this.element.onpause=null;this.element.onvolumechange=null;
  }
  this.element=null;this.notify('off');
 }
}

/** Opt-in compositor diagnostics. Element clocks need not share an origin,
 * so this frame delta is not an audio/video synchronization measurement. */
export function observePlayback(video:HTMLVideoElement,audio:HTMLAudioElement){
 let closed=false,callback=0,samples=0;
 let baselineTime:number|null=null,baselineFrames=0;
 function reset(){
  samples=0;baselineTime=null;baselineFrames=0;
  video.dataset.playbackProbe=JSON.stringify({state:'waiting',samples:0,elementFrameDeltaMs:null});
 }
 if(typeof video.requestVideoFrameCallback!=='function'){
  video.dataset.playbackProbe=JSON.stringify({state:'unsupported',samples:0,elementFrameDeltaMs:null});
  return()=>{delete video.dataset.playbackProbe};
 }
 function frame(now:number,metadata:VideoFrameCallbackMetadata){
  if(closed)return;
  if(video.readyState>=3&&!video.paused&&audio.readyState>=3&&!audio.paused&&!audio.muted&&audio.volume>0){
   const audioAtDisplay=audio.currentTime+(metadata.expectedDisplayTime-now)/1000*audio.playbackRate;
   const offset=(metadata.mediaTime-audioAtDisplay)*1000;
   samples++;
   if(baselineTime===null){baselineTime=metadata.expectedDisplayTime;baselineFrames=metadata.presentedFrames}
   video.dataset.playbackProbe=JSON.stringify({state:'playing',samples,elementFrameDeltaMs:offset,
    presentedFrames:metadata.presentedFrames,
    renderedFps:metadata.expectedDisplayTime>baselineTime?
     (metadata.presentedFrames-baselineFrames)*1000/(metadata.expectedDisplayTime-baselineTime):null});
  }else{
   video.dataset.playbackProbe=JSON.stringify({state:'waiting',samples,elementFrameDeltaMs:null});
  }
  callback=video.requestVideoFrameCallback(frame);
 }
 reset();video.addEventListener('emptied',reset);audio.addEventListener('emptied',reset);
 callback=video.requestVideoFrameCallback(frame);
 return()=>{
  closed=true;video.cancelVideoFrameCallback(callback);
  video.removeEventListener('emptied',reset);audio.removeEventListener('emptied',reset);
  delete video.dataset.playbackProbe;
 };
}

/** Compare source NTP clocks, as defined by WebRTC's statistics specification.
 * Missing browser support remains explicit rather than being treated as zero. */
export function observeRtcSynchronization(peer:RTCPeerConnection,stream:MediaStream,video:HTMLVideoElement,audioElement:HTMLAudioElement){
 let closed=false,timer:ReturnType<typeof setTimeout>|undefined,samples=0,peak=0;
 const offsets:number[]=[];
 type PlayoutStats=RTCInboundRtpStreamStats&{estimatedPlayoutTimestamp?:number};
 async function sample(){
  try{
   const stats=await peer.getStats();if(closed)return;
   const ids=new Set(stream.getTracks().map(track=>track.id));
   let audio:PlayoutStats|undefined,picture:PlayoutStats|undefined;
   stats.forEach(stat=>{
    if(stat.type!=='inbound-rtp'||!ids.has(stat.trackIdentifier))return;
    if(stat.kind==='audio')audio=stat;else if(stat.kind==='video')picture=stat;
   });
   const a=audio?.estimatedPlayoutTimestamp,v=picture?.estimatedPlayoutTimestamp;
   const playing=video.readyState>=3&&!video.paused&&audioElement.readyState>=3&&!audioElement.paused&&!audioElement.muted&&audioElement.volume>0;
   const measured=playing&&Number.isFinite(a)&&Number.isFinite(v)&&a!>0&&v!>0;
   const offset=measured?a!-v!:null;
   if(offset!==null){samples++;peak=Math.max(peak,Math.abs(offset));offsets.push(Math.abs(offset));if(offsets.length>1200)offsets.shift()}
   const ordered=[...offsets].sort((a,b)=>a-b);
   video.dataset.rtcSynchronization=JSON.stringify({state:!playing?'waiting':measured?'measured':'unavailable',samples,
    latestOffsetMs:offset,maxAbsOffsetMs:samples?peak:null,
    p95AbsOffsetMs:samples?ordered[Math.ceil(ordered.length*.95)-1]:null,
    audio:audio&&{estimatedPlayoutTimestamp:a,jitterBufferDelay:audio.jitterBufferDelay,jitterBufferEmittedCount:audio.jitterBufferEmittedCount,concealedSamples:audio.concealedSamples},
    video:picture&&{estimatedPlayoutTimestamp:v,jitterBufferDelay:picture.jitterBufferDelay,jitterBufferEmittedCount:picture.jitterBufferEmittedCount,framesDecoded:picture.framesDecoded,framesDropped:picture.framesDropped,framesPerSecond:picture.framesPerSecond}});
  }catch{if(!closed)video.dataset.rtcSynchronization=JSON.stringify({state:'unavailable',samples,latestOffsetMs:null})}
  if(!closed)timer=setTimeout(()=>void sample(),500);
 }
 void sample();
 return()=>{closed=true;clearTimeout(timer);delete video.dataset.rtcSynchronization};
}
