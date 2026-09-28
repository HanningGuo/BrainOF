'use strict';
const $ = (q, el = document) => el.querySelector(q);
const $$ = (q, el = document) => [...el.querySelectorAll(q)];
const config = window.BRAIN_OF_CONFIG;
const fmt = n => n.toLocaleString('en-US');
const colors = {fMRI:'#4263d4', EEG:'#b67215', MEG:'#178477'};
const safeUrl = value => {try {const u = new URL(value, location.href); return ['http:','https:'].includes(u.protocol) ? u.href : null;} catch {return null;}};
const el = (tag, text, cls) => {const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;};
// Unreleased resources are explicitly disabled; no fabricated destinations.
$$('[data-resource-buttons]').forEach(container => {
  const items = [['arxiv','arXiv'],['github','Code'],['huggingface','Checkpoints']];
  for(const [key,label] of items){const url=config.links[key] && safeUrl(config.links[key]);const item=el(url?'a':'button',undefined,'resource-button resource-'+key+(url?' live':''));const icon=el('img');icon.src='assets/resource-'+key+'.svg';icon.alt='';icon.width=24;icon.height=24;icon.className='resource-icon';item.append(icon,el('span',label));if(url){item.href=url;item.target='_blank';item.rel='noopener noreferrer';item.append(el('span','↗'));}else{item.disabled=true;item.title=label+' link will be added when released';}container.append(item);}
  for(const entry of config.links.additional || []){const url=safeUrl(entry.url);if(url){const a=el('a',entry.label,'resource-button live');a.href=url;a.target='_blank';a.rel='noopener noreferrer';container.append(a);}}
});
$('#citation').textContent=config.citation;
$('#copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(config.citation);$('#copy-status').textContent='BibTeX copied.';}catch{$('#copy-status').textContent='Select the citation text below to copy it.';} });
// Optional venue and persistent navigation shortcut.
if(config.venue?.name){$('#venue').hidden=false;$('#venue').textContent=config.venue.name+(config.venue.status?' · '+config.venue.status:'');}
$('#persistent-resources').hidden=config.showPersistentResources===false;
// Model cards are driven by configuration; adding or releasing Giant requires no HTML edits.
const models=config.models||[];
for(const model of models){
 const planned=model.status==='planned';const card=el('article',undefined,'model-card'+(model.featured?' model-featured':'')+(planned?' model-planned':''));
 card.append(el('span',model.name.toUpperCase(),'model-size'));
 const heading=el('h3',planned?'In development':model.total);if(!planned)heading.append(el('span',model.unit));card.append(heading,el('p',planned?'Pretraining is planned':'total parameters'));
 if(!planned){const dl=el('dl');for(const [label,value] of [['Active parameters',model.active],['Backbone layers',model.layers]]){const row=el('div');row.append(el('dt',label),el('dd',String(value)));dl.append(row);}card.append(dl);}
 else card.append(el('p','Scale, evaluation results, and checkpoints will follow.','planned-description'));
 const checkpoint=model.checkpoint||(model.status==='released'?config.links.huggingface:null);const url=checkpoint&&safeUrl(checkpoint);const release=el(url?'a':'span',url?'Explore checkpoint ↗':planned?'Planned variant':'Checkpoint link forthcoming','release-label');if(url){release.href=url;release.target='_blank';release.rel='noopener noreferrer';}card.append(release);$('#model-grid').append(card);
}
if(config.awards?.length){const region=$('#awards');region.hidden=false;for(const award of config.awards){const url=award.url&&safeUrl(award.url);const a=el(url?'a':'span',`${award.result} · ${award.title}${award.date?' · '+award.date:''}`,'award');if(url)a.href=url;region.append(a);}}
if(config.video?.src && safeUrl(config.video.src)){const region=$('#video-section');region.hidden=false;region.append(el('h3',config.video.title||'Research overview'));const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=safeUrl(config.video.src);if(config.video.poster&&safeUrl(config.video.poster))video.poster=safeUrl(config.video.poster);if(config.video.captions&&safeUrl(config.video.captions)){const track=document.createElement('track');track.kind='captions';track.srclang='en';track.label='English';track.src=safeUrl(config.video.captions);video.append(track);}region.append(video);}
// Contents stays available at every width; the drawer and desktop rail share state.
const contentsButton = $('.menu-button');
const contentsSidebar = $('#sidebar');
const compactNavigation = matchMedia('(max-width: 900px)');
let desktopContentsOpen = true;
let drawerContentsOpen = false;
function renderContents() {
 const open = compactNavigation.matches ? drawerContentsOpen : desktopContentsOpen;
 contentsSidebar.classList.toggle('open', open);
 document.body.classList.toggle('contents-collapsed', !compactNavigation.matches && !open);
 contentsSidebar.inert = !open;
 contentsSidebar.setAttribute('aria-hidden', String(!open));
 contentsButton.setAttribute('aria-expanded', String(open));
 contentsButton.setAttribute('aria-label', (open ? 'Hide' : 'Show') + ' page contents');
}
function closeContents(restoreFocus = false) {
 if (compactNavigation.matches) drawerContentsOpen = false;
 else desktopContentsOpen = false;
 if (restoreFocus || contentsSidebar.contains(document.activeElement)) contentsButton.focus();
 renderContents();
}
contentsButton.addEventListener('click', () => {
 if (compactNavigation.matches) drawerContentsOpen = !drawerContentsOpen;
 else desktopContentsOpen = !desktopContentsOpen;
 renderContents();
});
$$('#sidebar a').forEach(a => a.addEventListener('click', () => {
 if (!compactNavigation.matches) return;
 const target = document.getElementById(a.hash.slice(1));
 if (target) { target.setAttribute('tabindex', '-1'); target.focus({preventScroll: true}); }
 closeContents();
}));
document.addEventListener('keydown', e => {
 if (e.key === 'Escape' && (drawerContentsOpen || contentsSidebar.contains(document.activeElement))) closeContents(true);
});
document.addEventListener('click', e => {
 if (compactNavigation.matches && drawerContentsOpen && !contentsSidebar.contains(e.target) && !contentsButton.contains(e.target)) closeContents();
});
compactNavigation.addEventListener('change', () => {
 drawerContentsOpen = false;
 if (compactNavigation.matches && contentsSidebar.contains(document.activeElement)) contentsButton.focus();
 renderContents();
});
renderContents();
const sections=$$('main > section');
let scheduled=false;
function updateNav(){let current=sections[0].id;for(const s of sections)if(s.getBoundingClientRect().top<180)current=s.id;$$('#sidebar nav a').forEach(a=>{const active=a.hash==='#'+current;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});scheduled=false;}
addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(updateNav);}},{passive:true});updateNav();
const dialog=$('#figure-dialog');
$$('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{$('#enlarged-figure').src=button.dataset.zoom;$('#enlarged-figure').alt=button.dataset.caption;$('#figure-caption').textContent=button.dataset.caption;dialog.showModal();}));
$('#close-figure').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
const recon={eeg:['Temporal dynamics & spectral bursts','Compare the original and reconstructed EEG waveform and spectrogram. These qualitative examples show how temporal and spectral structure are represented together.','EEG signal and spectrogram'],fmri:['Signals & functional connectivity','Compare the original and reconstructed BOLD signal alongside the corresponding functional-connectivity matrices. The example examines whether inter-regional structure is preserved.','fMRI signal and functional connectivity'],meg:['Signals & sensor topography','Compare the original and reconstructed MEG signal and sensor topomaps. The example examines the recovery of spatial magnetic-field patterns.','MEG signal and sensor topography']};
$$('[data-recon]').forEach(b=>b.addEventListener('click',()=>{const k=b.dataset.recon;$$('[data-recon]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));$('#recon-title').textContent=recon[k][0];$('#recon-description').textContent=recon[k][1];$('#recon-img').src='assets/reconstruction-'+k+'.png';$('#recon-img').alt='Original and reconstructed '+recon[k][2];$('#recon-zoom').dataset.zoom=$('#recon-img').getAttribute('src');$('#recon-zoom').dataset.caption=recon[k][2];}));
const benchmarks=[['SEED-V','EEG','Emotion recognition','Cohen’s κ (%) ↑',false],['CHB-MIT','EEG','Seizure detection','Balanced accuracy (%) ↑',false],['TUAB','EEG','Abnormality detection','Balanced accuracy (%) ↑',false],['ADNI','fMRI','Alzheimer’s disease classification','Balanced accuracy (%) ↑',false],['ADHD-200','fMRI','ADHD identification','Balanced accuracy (%) ↑',false],['CamCAN','fMRI','Brain-age prediction','Mean absolute error (years) ↓',true],['CamCAN','MEG','Brain-age prediction','Mean absolute error (years) ↓',true],['LEMON','fMRI','Age-group classification','Balanced accuracy (%) ↑',false],['LEMON','EEG','Age-group classification','Balanced accuracy (%) ↑',false]];
benchmarks.forEach((b,i)=>{const o=el('option',`${b[0]} · ${b[1]} — ${b[2]}`);o.value=i;$('#benchmark').append(o);});$('#benchmark').value='2';
let results=[];
function renderResults(){const index=Number($('#benchmark').value);const b=benchmarks[index];const bases=results.filter(r=>!r.name.startsWith('Brain-OF'));const best=bases.reduce((a,c)=>(b[4]?c.values[index][0]<a.values[index][0]:c.values[index][0]>a.values[index][0])?c:a);const ours=results.filter(r=>r.name.startsWith('Brain-OF'));const subset=[{...best,baseline:true},...ours];const max=b[4]?Math.ceil(Math.max(...subset.map(r=>r.values[index][0]))*1.25):100;
 $('#result-modality').textContent=b[1]+' · '+b[0];$('#result-title').textContent=b[2];$('#result-metric').textContent=b[3]+(b[4]?' · Lower is better':' · Higher is better');$('#result-bars').replaceChildren();
 subset.forEach(r=>{const row=el('div',undefined,'bar-row'+(r.baseline?' baseline':'')+(r.name.endsWith('Huge')?' huge':''));const name=el('span',r.baseline?r.name+' (baseline)':r.name);const track=el('div',undefined,'bar-track');const bar=el('div',undefined,'bar-fill');bar.style.width=(r.values[index][0]/max*100)+'%';track.append(bar);row.append(name,track,el('span',r.values[index].map(n=>n.toFixed(2)).join(' ± '),'bar-value'));$('#result-bars').append(row);});
 const huge=ours.find(r=>r.name.endsWith('Huge'));const difference=(huge.values[index][0]-best.values[index][0])*(b[4]?-1:1);$('#result-takeaway').textContent=difference>0?`Brain-OF Huge ${b[4]?'reduces MAE by':'improves the score by'} ${Math.abs(difference).toFixed(2)} ${b[4]?'years':'percentage points'} compared with the strongest reported baseline (${best.name}) in this setting.`:`The strongest reported baseline (${best.name}) remains ahead of Brain-OF Huge by ${Math.abs(difference).toFixed(2)} ${b[4]?'years MAE':'percentage points'} in this setting.`;
 $('#all-results-summary').textContent='Compare all methods · '+b[0]+' · '+b[1];$('#all-results-caption').textContent=b[0]+' · '+b[1]+' · '+b[3];const body=$('#all-results tbody');body.replaceChildren();results.forEach(r=>{const tr=el('tr',undefined,r.name.startsWith('Brain-OF')?'ours':'');tr.append(el('td',r.name),el('td',r.values[index].map(n=>n.toFixed(2)).join(' ± ')));body.append(tr);});
}
$('#benchmark').addEventListener('change',()=>{if(results.length)renderResults();});
const fusion={CamCAN:{metric:'MAE in years ↓',labels:['fMRI','MEG','fMRI + MEG'],data:[[[12.09,.78],[8.99,.14],[8.87,.09]],[[11.71,.67],[8.86,.21],[8.78,.07]],[[9.51,.51],[7.87,.22],[8.21,.11]]]},LEMON:{metric:'Balanced accuracy (%) ↑',labels:['fMRI','EEG','fMRI + EEG'],data:[[[58.95,2.13],[76.05,.44],[76.77,.60]],[[62.85,4.29],[78.04,1.59],[78.32,1.64]],[[61.79,1.29],[80.39,.59],[81.10,.54]]]}};
function renderFusion(i){$('#fusion-panels').replaceChildren();for(const [name,d] of Object.entries(fusion)){const card=el('article',undefined,'fusion-panel');card.append(el('h3',name),el('p',d.metric,'small muted'));const values=d.data[i];const best=name==='CamCAN'?Math.min(...values.map(v=>v[0])):Math.max(...values.map(v=>v[0]));values.forEach((v,j)=>{const row=el('div',undefined,'fusion-row'+(v[0]===best?' best':''));const value=el('strong',v[0].toFixed(2)+' ');value.append(el('small','± '+v[1].toFixed(2)));row.append(el('span',d.labels[j]+(j===2?' · Fusion':'')),value);card.append(row);});$('#fusion-panels').append(card);}$('#fusion-note').textContent=i===2?'Fusion improves LEMON age-group classification, but not CamCAN age prediction at Huge scale: MEG alone achieves 7.87 years MAE versus 8.21 for fusion.':'At this scale, fusion improves the mean score over the strongest unimodal result on both paired benchmarks. Gains are modest and vary with task and model size.';}
$$('[data-fusion]').forEach(b=>b.addEventListener('click',()=>{$$('[data-fusion]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderFusion(Number(b.dataset.fusion));}));renderFusion(0);
let datasets=[];
function renderDatasets(mod){const chart=$('#dataset-chart');chart.replaceChildren();const axis=el('div',undefined,'log-axis');axis.append(el('span','Samples →'));const ticks=el('div',undefined,'log-ticks');['100','1K','10K','100K','1M','10M'].forEach(t=>ticks.append(el('span',t)));axis.append(ticks,el('span',''));chart.append(axis);datasets.filter(r=>r.modality===mod).sort((a,b)=>b.samples-a.samples).forEach(r=>{const row=el('div',undefined,'dataset-row');row.append(el('div',r.name));const track=el('div',undefined,'dataset-bar-track');const bar=el('div',undefined,'dataset-bar');bar.style.width=((Math.log10(r.samples)-2)/5*100)+'%';bar.style.background=colors[mod];track.append(bar);row.append(track,el('span',fmt(r.samples),'dataset-count'));row.title=`${r.name}: ${fmt(r.samples)} samples; ${fmt(r.participants)} participants; ${r.paradigm}`;chart.append(row);});}
$$('[data-dataset]').forEach(b=>b.addEventListener('click',()=>{$$('[data-dataset]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));if(datasets.length)renderDatasets(b.dataset.dataset);}));
async function loadData(){const settled=await Promise.allSettled([fetch('assets/results.json').then(r=>{if(!r.ok)throw Error();return r.json();}),fetch('assets/datasets.json').then(r=>{if(!r.ok)throw Error();return r.json();})]);
 if(settled[0].status==='fulfilled'){results=settled[0].value;renderResults();}else{$('#result-bars').textContent='Results could not be loaded. Please refresh the page.';}
 if(settled[1].status==='fulfilled'){datasets=settled[1].value;
 const modalityDetails={fMRI:['Blood-oxygenation changes across brain regions','116–246 ROIs'],EEG:['Electrical activity recorded at the scalp','17–64 channels'],MEG:['Magnetic fields generated by neural activity','248–306 sensors']};
 for(const mod of ['fMRI','EEG','MEG']){const subset=datasets.filter(r=>r.modality===mod);const n=subset.reduce((a,r)=>a+r.samples,0);const card=el('article',undefined,'corpus-profile');card.style.setProperty('--modality-color',colors[mod]);card.append(el('h3',mod),el('p',modalityDetails[mod][0],'profile-description'));const count=el('strong',fmt(n),'profile-total');card.append(count,el('span','pretraining samples','profile-unit'));const facts=el('dl');for(const [label,value] of [['Dataset entries',subset.length],[mod==='fMRI'?'Brain parcellation':'Recording channels',modalityDetails[mod][1]]]){const row=el('div');row.append(el('dt',label),el('dd',String(value)));facts.append(row);}card.append(facts);$('#corpus-profiles').append(card);}
 renderDatasets('fMRI');const tbody=$('#dataset-table tbody');datasets.forEach(r=>{const tr=el('tr');[r.name,r.modality,r.paradigm,r.sr,fmt(r.channels),fmt(r.participants),fmt(r.samples)].forEach(t=>tr.append(el('td',t)));tbody.append(tr);});}else{$('#dataset-chart').textContent='Dataset details could not be loaded. Please refresh the page.';}
}
loadData();
