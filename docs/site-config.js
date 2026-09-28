window.BRAIN_OF_CONFIG = {
  links: { arxiv: null, github: 'https://github.com/HanningGuo/BrainOF', huggingface: null, additional: [] },
  venue: { name: 'NeurIPS 2026' },
  showPersistentResources: true,
  models: [
    { name: 'Base', total: '47.5', unit: 'M', active: '21.5M', layers: 12, status: 'reported', checkpoint: null },
    { name: 'Large', total: '331', unit: 'M', active: '150M', layers: 24, status: 'reported', checkpoint: null },
    { name: 'Huge', total: '1.7', unit: 'B', active: '500M', layers: 36, status: 'reported', checkpoint: null },
    { name: 'Giant', total: '9', unit: ' B', active: '1.2B', layers: 48, status: 'reported', checkpoint: null, featured: true }
  ],
  // Example: { src: 'assets/talk.mp4', poster: 'assets/talk-poster.jpg', title: 'Brain-OF overview', captions: 'assets/talk.vtt' }
  video: null,
  // Add confirmed results only: { title: 'Competition name', result: 'First place', date: '2027', url: 'https://...' }
  awards: [],
  citation: `@article{guo2026brainof,
  title = {Brain-OF: An Omnifunctional Foundation Model for fMRI, EEG and MEG},
  author = {Guo, Hanning and Bi, Hanwen and Abdellatif, Farah and Galbenus, Andrei and Shah, Jon. N. and Morrison, Abigail and Dammers, Jürgen},
  journal = {Advances in Neural Information Processing Systems},
  year = {2026}
}`
};
