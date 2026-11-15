import * as THREE from "three";

// Physically-motivated Earth: Blue Marble by day, city lights by night,
// soft terminator, Lambert shading toward the Sun and specular sun glint on
// the oceans (ocean mask).  World-space normals: the globe mesh is static and
// the camera orbits it, so the Sun direction is a plain world vector.
export function makeEarthMaterial(): { material: THREE.ShaderMaterial; setSun: (dir: THREE.Vector3) => void } {
  const loader = new THREE.TextureLoader();
  const tex = (url: string, srgb = true) => {
    const t = loader.load(url);
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    return t;
  };
  const material = new THREE.ShaderMaterial({
    uniforms: {
      dayTexture: { value: tex("/textures/earth-blue-marble.jpg") },
      nightTexture: { value: tex("/textures/earth-night.jpg") },
      oceanMask: { value: tex("/textures/earth-water.jpg", false) },
      sunDirection: { value: new THREE.Vector3(1, 0, 0) },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vPosW;
      void main() {
        vUv = uv;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vPosW = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D dayTexture;
      uniform sampler2D nightTexture;
      uniform sampler2D oceanMask;
      uniform vec3 sunDirection;
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vPosW;
      void main() {
        vec3 n = normalize(vNormalW);
        vec3 s = normalize(sunDirection);
        float cosI = dot(n, s);
        vec3 day = texture2D(dayTexture, vUv).rgb;
        vec3 night = texture2D(nightTexture, vUv).rgb;
        float ocean = texture2D(oceanMask, vUv).r;
        // Lambert + a little ambient on the day side
        vec3 lit = day * (0.10 + 1.05 * clamp(cosI, 0.0, 1.0));
        // sun glint on water
        vec3 v = normalize(cameraPosition - vPosW);
        vec3 h = normalize(s + v);
        // sun glint on water, suppressed at grazing view angles (limb)
        float spec = pow(max(dot(n, h), 0.0), 140.0) * ocean
                     * smoothstep(0.05, 0.3, cosI) * smoothstep(0.15, 0.5, dot(n, v));
        lit += vec3(1.0, 0.95, 0.85) * spec * 0.45;
        // warm city lights on the night side, fading through twilight
        vec3 lights = pow(night, vec3(1.6)) * vec3(1.35, 1.05, 0.7) * 1.6;
        float dayMix = smoothstep(-0.12, 0.18, cosI);
        vec3 col = mix(lights + day * 0.015, lit, dayMix);
        // reddish twilight band at the terminator
        float twilight = smoothstep(-0.12, 0.0, cosI) * (1.0 - smoothstep(0.0, 0.12, cosI));
        col += vec3(0.07, 0.03, 0.012) * twilight * (1.0 - ocean * 0.5);
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  return { material, setSun: (dir) => material.uniforms.sunDirection.value.copy(dir).normalize() };
}

export function makeClouds(radius: number): { mesh: THREE.Mesh; setSun: (dir: THREE.Vector3) => void } {
  const t = new THREE.TextureLoader().load("/textures/earth-clouds.jpg");
  t.anisotropy = 8;
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { density: { value: t }, sunDirection: { value: new THREE.Vector3(1, 0, 0) } },
    vertexShader: /* glsl */ `
      varying vec2 vUv; varying vec3 vNormalW;
      void main() { vUv = uv; vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D density; uniform vec3 sunDirection;
      varying vec2 vUv; varying vec3 vNormalW;
      void main() {
        float d = texture2D(density, vUv).r;
        float cosI = dot(normalize(vNormalW), normalize(sunDirection));
        float light = 0.04 + 0.96 * smoothstep(-0.08, 0.25, cosI);
        gl_FragColor = vec4(vec3(light), d * 0.85 * (0.35 + 0.65 * light));
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.007, 96, 64), material);
  // three-globe's sphere is rotated so that u=0 sits at lng -180: match it
  mesh.rotation.y = -Math.PI / 2;
  return { mesh, setSun: (dir) => material.uniforms.sunDirection.value.copy(dir).normalize() };
}

export function makeSpacecraft(accent: string): THREE.Group {
  const g = new THREE.Group();
  const foil = new THREE.MeshPhongMaterial({ color: "#c8a24a", specular: "#fff1c1", shininess: 80, emissive: "#3a2a08" });
  const white = new THREE.MeshPhongMaterial({ color: "#d9dee7", shininess: 40, emissive: "#1c1f26" });
  const cell = new THREE.MeshPhongMaterial({ color: "#16264f", specular: "#7aa2ff", shininess: 120, emissive: "#0a1330", side: THREE.DoubleSide });
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 2.0), foil);
  g.add(body);
  const instrument = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.45, 0.7, 16), white);
  instrument.rotation.x = Math.PI / 2;
  instrument.position.set(0, -0.75, 0.4);
  g.add(instrument);
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.6, 6), white);
  boom.rotation.z = Math.PI / 2;
  boom.position.set(1.4, 0, -0.4);
  g.add(boom);
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 1.4, 6, 2), cell);
  panel.position.set(3.9, 0, -0.4);
  g.add(panel);
  const grid = new THREE.LineSegments(new THREE.WireframeGeometry(panel.geometry), new THREE.LineBasicMaterial({ color: "#5f7fbf", transparent: true, opacity: 0.6 }));
  grid.position.copy(panel.position);
  g.add(grid);
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), new THREE.MeshBasicMaterial({ color: accent }));
  beacon.position.set(0, 0.75, -0.9);
  g.add(beacon);
  g.scale.setScalar(1.6);
  return g;
}
