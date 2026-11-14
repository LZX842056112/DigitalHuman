export async function api(path:string, init?:RequestInit){
 const response=await fetch('/api'+path,init);
 if(!response.ok){let message=await response.text();try{message=JSON.parse(message).detail}catch{}throw Error(String(message))}
 return response.json();
}
export const json=(value:unknown)=>({headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
export function fileUrl(id:string,path:string){return '/api/avatars/'+encodeURIComponent(id)+'/files/'+path.split('/').filter(Boolean).map(encodeURIComponent).join('/')}
