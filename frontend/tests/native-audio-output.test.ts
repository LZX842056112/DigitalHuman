import test from 'node:test';
import assert from 'node:assert/strict';
import {RemoteAudioOutput} from '../src/native-audio-output.ts';

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
