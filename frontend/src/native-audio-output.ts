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
