function A(e,a,t,{mobile:m=!1}={}){let n=m?512:1024,i=14,r=new e.WebGLRenderTarget(n,n),c=new e.WebGLRenderTarget(n,n,{depthBuffer:!1});r.texture.generateMipmaps=c.texture.generateMipmaps=!1;let u=new e.OrthographicCamera(-i/2,i/2,i/2,-i/2,.001,2.5);u.position.set(.75,-.012,.25),u.up.set(0,0,1),u.lookAt(.75,1,.25);let w=new e.ShaderMaterial({side:e.DoubleSide,vertexShader:`
      varying float heightAboveFloor;
      void main() {
        vec4 world = modelMatrix * vec4(position, 1.0);
        heightAboveFloor = max(0.0, world.y);
        gl_Position = projectionMatrix * viewMatrix * world;
      }`,fragmentShader:`
      varying float heightAboveFloor;
      void main() {
        float contact = exp(-heightAboveFloor * 2.8);
        gl_FragColor = vec4(0.0, 0.0, 0.0, contact);
      }`}),l=new e.ShaderMaterial({depthTest:!1,depthWrite:!1,uniforms:{source:{value:r.texture},stepSize:{value:new e.Vector2}},vertexShader:`varying vec2 uvCoord;
      void main() { uvCoord = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,fragmentShader:`
      uniform sampler2D source;
      uniform vec2 stepSize;
      varying vec2 uvCoord;
      void main() {
        float alpha = 0.0;
        float total = 0.0;
        for (int i = -8; i <= 8; i++) {
          float weight = exp(-float(i * i) / 32.0);
          alpha += texture2D(source, uvCoord + float(i) * stepSize).a * weight;
          total += weight;
        }
        gl_FragColor = vec4(0.0, 0.0, 0.0, alpha / total);
      }`}),p=new e.Mesh(new e.PlaneGeometry(2,2),l);p.frustumCulled=!1;let d=new e.Scene;d.add(p);let h=new e.Camera,o=new e.Mesh(new e.PlaneGeometry(i,i),new e.MeshBasicMaterial({map:r.texture,transparent:!0,opacity:.4,depthWrite:!1,side:e.DoubleSide,toneMapped:!1}));o.rotation.x=-Math.PI/2,o.scale.y=-1,o.position.set(.75,-.01,.25),t.add(o);let g=new e.Color;return{plane:o,update(v=[]){let b=t.background,C=t.overrideMaterial,M=a.getRenderTarget(),x=a.shadowMap.enabled,S=a.getClearAlpha();a.getClearColor(g);let y=v.map(s=>s.visible);o.visible=!1,v.forEach(s=>{s.visible=!1}),t.background=null,t.overrideMaterial=w,a.shadowMap.enabled=!1,a.setClearColor(0,0),a.setRenderTarget(r),a.render(t,u);let f=.00175;l.uniforms.source.value=r.texture,l.uniforms.stepSize.value.set(f,0),a.setRenderTarget(c),a.render(d,h),l.uniforms.source.value=c.texture,l.uniforms.stepSize.value.set(0,f),a.setRenderTarget(r),a.render(d,h),a.setRenderTarget(M),a.setClearColor(g,S),a.shadowMap.enabled=x,t.background=b,t.overrideMaterial=C,o.visible=!0,v.forEach((s,z)=>{s.visible=y[z]})}}}export{A as createContactShadows};
