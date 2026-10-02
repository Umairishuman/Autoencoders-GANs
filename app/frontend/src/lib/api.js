import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: BASE });

export async function healthCheck() {
  const { data } = await api.get('/api/health');
  return data;
}

export async function getModels() {
  const { data } = await api.get('/api/models');
  return data;
}

export async function getSamples(kind = 'pets') {
  const { data } = await api.get('/api/samples', { params: { kind } });
  return data;
}

function buildForm(file, sample, params) {
  const fd = new FormData();
  if (file) fd.append('file', file);
  else if (sample) fd.append('sample', sample);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== '') fd.append(k, v);
  });
  return fd;
}

export async function corruptImage(file, sample, params = {}) {
  const fd = buildForm(file, sample, params);
  const { data } = await api.post('/api/corrupt', fd);
  return data;
}

export async function restoreUniversal(file, sample, params = {}) {
  const fd = buildForm(file, sample, params);
  const { data } = await api.post('/api/restore/universal', fd);
  return data;
}

export async function restoreHard(file, sample, params = {}) {
  const fd = buildForm(file, sample, params);
  const { data } = await api.post('/api/restore/hard', fd);
  return data;
}

export async function restoreSoft(file, sample, params = {}) {
  const fd = buildForm(file, sample, params);
  const { data } = await api.post('/api/restore/soft', fd);
  return data;
}

export async function generateSketch(file, sample, style = '0') {
  const fd = new FormData();
  if (file) fd.append('file', file);
  else if (sample) fd.append('sample', sample);
  fd.append('style', style);
  const { data } = await api.post('/api/sketch', fd);
  return data;
}
