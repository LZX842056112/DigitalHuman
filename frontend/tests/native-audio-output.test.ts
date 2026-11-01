import test from 'node:test';
import assert from 'node:assert/strict';
import {RemoteAudioOutput} from '../src/native-audio-output.ts';
import * as nativeMedia from '../src/native-audio-output.ts';

class FakeAudio {
 srcObject:MediaStream|null=null;volume=1;muted=false;paused=true;
 onplaying:(()=>void)|null=null;onpause:(()=>void)|null=null;onvolumechange:(()=>void)|null=null;
 played=0;fail=false;
 async play(){this.played++;if(this.fail)throw new Error('NotAllowedError');this.paused=false;this.onplaying?.()}
 pause(){this.paused=true;this.onpause?.()}
}

test('remote audio uses a real media element and waits for its stream',async()=>{
 const element=new FakeAudio();const states:string[]=[];
 const output=new RemoteAudioOutput(()=>element as unknown as HTMLAudioElement,(state)=>states.push(state));
 await output.enable();assert.equal(element.played,0);
 const stream={} as MediaStream;output.attach(stream);await Promise.resolve();
 assert.equal(element.srcObject,stream);assert.equal(element.played,1);assert.equal(states.at(-1),'running');
 output.setVolume(.4);assert.equal(element.volume,.4);
 await output.close();assert.equal(element.srcObject,null);assert.equal(element.paused,true);
});

test('a user click retries blocked native playback synchronously',async()=>{
 const element=new FakeAudio();element.fail=true;const states:string[]=[];
 const output=new RemoteAudioOutput(()=>element as unknown as HTMLAudioElement,(state)=>states.push(state));
 output.attach({} as MediaStream);await Promise.resolve();await Promise.resolve();
 assert.equal(states.at(-1),'blocked');element.fail=false;element.muted=true;
 const pending=output.enable();assert.equal(element.played,2);assert.equal(element.muted,false);
 await pending;assert.equal(states.at(-1),'running');await output.close();
});

test('replacing the remote stream leaves a single playback element',async()=>{
 const element=new FakeAudio();const output=new RemoteAudioOutput(()=>element as unknown as HTMLAudioElement,()=>{});
 const first={} as MediaStream;const second={} as MediaStream;
 output.attach(first);output.attach(second);await Promise.resolve();
 assert.equal(element.srcObject,second);await output.close();
});

class ProbeVideo extends EventTarget {
 dataset:Record<string,string>={};readyState=4;paused=false;
 callback:((now:number,metadata:VideoFrameCallbackMetadata)=>void)|null=null;
 next=0;cancelled:number[]=[];
 requestVideoFrameCallback(callback:(now:number,metadata:VideoFrameCallbackMetadata)=>void){this.callback=callback;return ++this.next}
 cancelVideoFrameCallback(id:number){this.cancelled.push(id)}
 present(now:number,mediaTime:number,presentedFrames:number){
  this.callback?.(now,{expectedDisplayTime:now+16,mediaTime,presentedFrames} as VideoFrameCallbackMetadata);
 }
}
class ProbeAudio extends EventTarget {
 currentTime=2;readyState=4;paused=false;muted=false;volume=1;playbackRate=1;
}

test('playback observation reports the element frame delta without calling it AV synchronization',()=>{
 assert.equal(typeof nativeMedia.observePlayback,'function');
 const video=new ProbeVideo(),audio=new ProbeAudio();
 const close=nativeMedia.observePlayback(video as unknown as HTMLVideoElement,audio as unknown as HTMLAudioElement);
 video.present(1000,2.016,1);audio.currentTime=2.04;video.present(1040,2.056,2);
 const report=JSON.parse(video.dataset.playbackProbe);
 assert.equal(report.state,'playing');assert.equal(report.samples,2);
 assert.ok(Math.abs(report.elementFrameDeltaMs)<.001);
 assert.equal(report.latestOffsetMs,undefined);
 assert.equal(report.renderedFps,25);
 close();assert.equal(video.dataset.playbackProbe,undefined);assert.ok(video.cancelled.length>0);
});

test('playback observation exposes late video instead of reporting false synchronization',()=>{
 assert.equal(typeof nativeMedia.observePlayback,'function');
 const video=new ProbeVideo(),audio=new ProbeAudio();audio.currentTime=10.3;
 const close=nativeMedia.observePlayback(video as unknown as HTMLVideoElement,audio as unknown as HTMLAudioElement);
 video.present(1000,10,1);
 const report=JSON.parse(video.dataset.playbackProbe);
 assert.ok(Math.abs(report.elementFrameDeltaMs+316)<.001);
 close();
});

test('paused or muted audio cannot produce a passing playback observation',()=>{
 assert.equal(typeof nativeMedia.observePlayback,'function');
 const video=new ProbeVideo(),audio=new ProbeAudio();audio.muted=true;
 const close=nativeMedia.observePlayback(video as unknown as HTMLVideoElement,audio as unknown as HTMLAudioElement);
 video.present(1000,2.016,1);
 const report=JSON.parse(video.dataset.playbackProbe);
 assert.equal(report.state,'waiting');assert.equal(report.samples,0);assert.equal(report.elementFrameDeltaMs,null);
 close();
});

test('RTC synchronization uses the two source-clock playout timestamps for the attached tracks',async(t)=>{
 assert.equal(typeof nativeMedia.observeRtcSynchronization,'function');
 const video=new ProbeVideo();
 const stats=new Map([
  ['a',{type:'inbound-rtp',kind:'audio',trackIdentifier:'voice',estimatedPlayoutTimestamp:100200}],
  ['v',{type:'inbound-rtp',kind:'video',trackIdentifier:'face',estimatedPlayoutTimestamp:100180}],
  ['old',{type:'inbound-rtp',kind:'video',trackIdentifier:'old',estimatedPlayoutTimestamp:90000}],
 ]);
 const pc={getStats:async()=>stats},stream={getTracks:()=>[{id:'voice'},{id:'face'}]};
 const close=nativeMedia.observeRtcSynchronization(pc as unknown as RTCPeerConnection,stream as unknown as MediaStream,video as unknown as HTMLVideoElement,new ProbeAudio() as unknown as HTMLAudioElement);
 t.after(close);
 await new Promise(resolve=>setImmediate(resolve));
 const report=JSON.parse(video.dataset.rtcSynchronization);
 assert.equal(report.state,'measured');assert.equal(report.latestOffsetMs,20);
 assert.equal(report.samples,1);assert.equal(report.maxAbsOffsetMs,20);
 close();assert.equal(video.dataset.rtcSynchronization,undefined);
});

test('missing RTC playout timestamps cannot produce a passing synchronization value',async(t)=>{
 assert.equal(typeof nativeMedia.observeRtcSynchronization,'function');
 const video=new ProbeVideo(),stats=new Map([
  ['a',{type:'inbound-rtp',kind:'audio',trackIdentifier:'voice'}],
  ['v',{type:'inbound-rtp',kind:'video',trackIdentifier:'face'}],
 ]);
 const close=nativeMedia.observeRtcSynchronization({getStats:async()=>stats} as unknown as RTCPeerConnection,
  {getTracks:()=>[{id:'voice'},{id:'face'}]} as unknown as MediaStream,video as unknown as HTMLVideoElement,new ProbeAudio() as unknown as HTMLAudioElement);
 t.after(close);
 await new Promise(resolve=>setImmediate(resolve));
 const report=JSON.parse(video.dataset.rtcSynchronization);
 assert.equal(report.state,'unavailable');assert.equal(report.latestOffsetMs,null);assert.equal(report.samples,0);
 close();
});

test('RTC timestamps cannot certify synchronization while native audio is muted',async(t)=>{
 const video=new ProbeVideo(),audio=new ProbeAudio();audio.muted=true;
 const stats=new Map([
  ['a',{type:'inbound-rtp',kind:'audio',trackIdentifier:'voice',estimatedPlayoutTimestamp:100200}],
  ['v',{type:'inbound-rtp',kind:'video',trackIdentifier:'face',estimatedPlayoutTimestamp:100180}],
 ]);
 const close=nativeMedia.observeRtcSynchronization({getStats:async()=>stats} as unknown as RTCPeerConnection,
  {getTracks:()=>[{id:'voice'},{id:'face'}]} as unknown as MediaStream,video as unknown as HTMLVideoElement,audio as unknown as HTMLAudioElement);
 t.after(close);
 await new Promise(resolve=>setImmediate(resolve));
 const report=JSON.parse(video.dataset.rtcSynchronization);
 assert.equal(report.state,'waiting');assert.equal(report.latestOffsetMs,null);assert.equal(report.samples,0);
 close();
});
