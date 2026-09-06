import fs from "node:fs"; import path from "node:path";
const root=new URL(".",import.meta.url).pathname;
const bad=[];
const req=["index.html","styles.css","script.js","data.js","favicon.svg","og.svg","manifest.webmanifest","robots.txt","sitemap.xml"];
for(const f of req) if(!fs.existsSync(path.join(root,f))) bad.push(`missing ${f}`);
const html=[];
const walk=d=>{for(const n of fs.readdirSync(d)){const p=path.join(d,n),s=fs.statSync(p);if(s.isDirectory()&&n!=="node_modules")walk(p);else if(n.endsWith(".html"))html.push(p)}};
walk(root);
for(const f of html){const t=fs.readFileSync(f,"utf8");if(!t.includes('id="app"'))bad.push(`missing app shell ${path.relative(root,f)}`);if(!t.includes("styles.css"))bad.push(`missing css ${path.relative(root,f)}`);if(!t.includes('type="application/ld+json"'))bad.push(`missing JSON-LD ${path.relative(root,f)}`)}
const skillsRoot=path.join(root,"content","skills");const skills=fs.existsSync(skillsRoot)?fs.readdirSync(skillsRoot).filter(x=>fs.statSync(path.join(skillsRoot,x)).isDirectory()):[];
if(skills.length!==10)bad.push(`expected 10 skill packages, found ${skills.length}`);
for(const slug of skills)if(!fs.existsSync(path.join(skillsRoot,slug,"SKILL.md")))bad.push(`missing SKILL.md ${slug}`);
const docSources=["getting-started","models","development","training","skills","reference"];for(const slug of docSources)if(!fs.existsSync(path.join(root,"content","docs",slug+".md")))bad.push(`missing doc source ${slug}`);
const researchSources=["architecture","tokenizer","training","evaluation","limitations"];for(const slug of researchSources)if(!fs.existsSync(path.join(root,"content","research",slug+".md")))bad.push(`missing research source ${slug}`);
const routes=["index.html","models/index.html","models/lapis-tiny/index.html","models/lapis-small/index.html","models/lapis-1b/index.html","models/lapis-3b/index.html","models/lapis-7b/index.html","skills/index.html","docs/index.html","research/index.html","roadmap/index.html","changelog/index.html","about/index.html","404.html"];
for(const f of routes)if(!fs.existsSync(path.join(root,f)))bad.push(`missing route ${f}`);
if(bad.length){console.error(bad.join("\n"));process.exit(1)}
console.log(`Lapis website validation passed: ${html.length} HTML pages; ${skills.length} skill packages.`);
