// npm install --no-save playwright@1.56.1; npx playwright install chromium
// Start the real app first. Requires ffmpeg/ffprobe.
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');const {pathToFileURL}=require('url');const {spawnSync}=require('child_process');
const repo=path.resolve(__dirname,'..'),raw=path.join(repo,'.demo-recording');fs.mkdirSync(raw,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true});const ctx=await browser.newContext({viewport:{width:1200,height:820},recordVideo:{dir:raw,size:{width:1200,height:820}}});const page=await ctx.newPage();
 const images=['benchmarks/segmentation_usefulness/examples.png','benchmarks/segmentation_usefulness/segmentation.png','benchmarks/super_fast/measured_performance.png'];
 for(const [i,p] of images.entries()){
  await page.goto(pathToFileURL(path.join(repo,p)).href);await page.waitForTimeout(i===0?2600:2500);
  await page.screenshot({path:path.join(raw,`frame-${i}.png`)});
 }
 const video=page.video();await ctx.close();const src=await video.path();await browser.close();
 const filter="[0:v]fps=12,scale=800:-1:flags=lanczos,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Saved measured MuSc outputs - not live inference':x=12:y=h-23:fontsize=13:fontcolor=white:box=1:boxcolor=black@0.8,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a";
 const dest=path.join(repo,'media/demo.gif');const r=spawnSync('ffmpeg',['-y','-i',src,'-filter_complex',filter,'-loop','0',dest],{encoding:'utf8'});if(r.status)throw Error(r.stderr.slice(-1000));
 fs.writeFileSync(path.join(repo,'media/DEMO.md'),'# Recorded MuSc library results\n\nFresh Playwright screen recording of the browser viewing the repositoryâ€™s actual saved benchmark output images, in this order:\n\n'+images.map(x=>'- ['+x+'](../'+x+')').join('\n')+'\n\nThis library has no desktop app. The GIF shows saved measured segmentations/overlays and performance results, not live inference or a new speed measurement. No result images, numbers or UI were fabricated. Benchmark methods, dataset attribution and limitations remain in the linked reports. Source browser video and screenshots stay in ignored .demo-recording/.\n');console.log(fs.statSync(dest).size);
})().catch(e=>{console.error(e);process.exitCode=1});
