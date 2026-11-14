import React from 'react';

type Props={
 sourceUrl:string|null;
 composeUrl:string|null;
 previewUrl:string|null;
 resultUrl:string|null;
 progress:number;
 stage:string;
 hint:string;
};

/** 原图 / 中间图 / 结果 three way comparison for the edit workbench. */
export default function EditPreview({sourceUrl,composeUrl,previewUrl,resultUrl,progress,stage,hint}:Props){
 const middle=previewUrl||composeUrl;
 return <div className="photo-edit-preview">
  <figure className="photo-edit-frame">
   <figcaption>原图</figcaption>
   {sourceUrl?<img src={sourceUrl} alt="原始照片"/>:<div className="photo-edit-blank">没有照片</div>}
  </figure>
  <figure className="photo-edit-frame">
   <figcaption>{previewUrl?'生成中':'构图预览'}</figcaption>
   {middle
    ?<img key={middle} src={middle} alt={previewUrl?'生成中的画面':'将要生成的画布'}/>
    :<div className="photo-edit-blank">{stage||'设置参数后会显示构图'}</div>}
   {progress>0&&progress<1&&<div className="photo-edit-progress"><progress max={1} value={progress}/><span>{Math.round(progress*100)}%</span></div>}
  </figure>
  <figure className="photo-edit-frame">
   <figcaption>结果</figcaption>
   {resultUrl?<img src={resultUrl} alt="编辑结果"/>:<div className="photo-edit-blank">完成后在这里查看</div>}
  </figure>
  <p className="photo-hint photo-edit-hint">{hint}</p>
 </div>;
}
