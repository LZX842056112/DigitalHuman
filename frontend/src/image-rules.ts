// Mirrors backend/image_rules.py. The server stays the only authority; this
// copy exists so an obviously wrong file never leaves the browser.
export type ImageRole='person'|'garment';

type Rule={
 formats:string[];suffixes:string[];minShort:number;maxEdge:number;aspect:[number,number];
 formatError:string;shortError:string;edgeError:string;aspectError:string;
};

export const IMAGE_RULES:Record<ImageRole,Rule>={
 person:{
  formats:['image/jpeg','image/png'],suffixes:['.jpg','.jpeg','.png'],
  minShort:320,maxEdge:8192,aspect:[0.33,3.0],
  formatError:'只支持 JPG 或 PNG 照片',
  shortError:'照片短边至少 320 像素，请换一张更清晰的照片',
  edgeError:'照片长边不能超过 8192 像素',
  aspectError:'照片长宽比异常，请上传单人照片',
 },
 garment:{
  formats:['image/jpeg','image/png','image/webp'],suffixes:['.jpg','.jpeg','.png','.webp'],
  minShort:256,maxEdge:4096,aspect:[0.5,2.0],
  formatError:'服装图只支持 JPG、PNG 或 WEBP',
  shortError:'服装图短边至少 256 像素',
  edgeError:'服装图长边不能超过 4096 像素',
  aspectError:'服装图长宽比异常，请上传单件衣服的正面图',
 },
};

function suffixOf(name:string){const dot=name.lastIndexOf('.');return dot<0?'':name.slice(dot).toLowerCase()}

async function measure(file:File):Promise<{width:number;height:number}>{
 if(typeof createImageBitmap==='function'){
  const bitmap=await createImageBitmap(file);
  const size={width:bitmap.width,height:bitmap.height};
  bitmap.close();
  return size;
 }
 return await new Promise((resolve,reject)=>{
  const url=URL.createObjectURL(file);
  const image=new Image();
  image.onload=()=>{resolve({width:image.naturalWidth,height:image.naturalHeight});URL.revokeObjectURL(url)};
  image.onerror=()=>{URL.revokeObjectURL(url);reject(Error('图片无法读取，可能已损坏'))};
  image.src=url;
 });
}

export async function checkImage(file:File,role:ImageRole):Promise<string|null>{
 const rule=IMAGE_RULES[role];
 const suffix=suffixOf(file.name);
 if(suffix&&!rule.suffixes.includes(suffix))return rule.formatError;
 const type=(file.type||'').toLowerCase();
 // Some browsers still report image/jpg for a plain JPEG.
 if(type&&type!=='image/jpg'&&!rule.formats.includes(type))return rule.formatError;
 let size:{width:number;height:number};
 try{size=await measure(file)}catch{return '图片无法读取，可能已损坏'}
 const long=Math.max(size.width,size.height),short=Math.min(size.width,size.height);
 if(long>rule.maxEdge)return rule.edgeError;
 if(short<rule.minShort)return rule.shortError;
 const ratio=size.height?size.width/size.height:0;
 if(ratio<rule.aspect[0]||ratio>rule.aspect[1])return rule.aspectError;
 return null;
}
