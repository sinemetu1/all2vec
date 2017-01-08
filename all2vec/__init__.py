"""Set of entity types storing high dimensional vectors."""

from collections import namedtuple
import logging
import os
import json
import dill
from annoy import AnnoyIndex

__all__ = ('EntitySet',)

logging.basicConfig(level=logging.INFO)

EntityVector = namedtuple("EntityVector", ["entity_id", "vector"])


class EntityType(object):
    def __init__(self, nfactor, ntrees, metric='angular',
                 entity_type_id=None, entity_type=None):
        """Initialize EntityType."""
        # metadata
        self._nfactor = nfactor
        self._metric = metric
        # object is accessed using this id. e.g. 'user'
        self._entity_type = entity_type
        # data is loaded in using this id. This can be more compact than the
        # entity_type, depending on the data source
        self._entity_type_id = entity_type_id
        self._ntrees = ntrees
        # data
        self._ann_obj = AnnoyIndex(nfactor, metric)
        # maps entity id to internal representation of id
        self._ann_map = {}
        # maps internal representation of id to entity id
        self._ann_map_inv = {}
        self._nitems = 0
    def add_item(self, entity_id, factors):
        """Add item, populating id map."""
        if entity_id in self._ann_map:
            raise ValueError('Duplicate entity: type = {0}, id = {1}'.format(
                self._entity_type, entity_id))
        self._ann_obj.add_item(self._nitems, factors)
        self._ann_map[entity_id] = self._nitems
        self._nitems = self._nitems + 1
    def build(self, verbose=False):
        """Build annoy model, create invert dictionary for future lookups."""
        self._ann_obj.verbose(verbose)
        self._ann_obj.build(self._ntrees)
        # this is only necessary after build, so we'll create it here
        self._ann_map_inv = {v: k for k, v in self._ann_map.items()}
    def get_nns_by_vector(self, vec, n, search_k):
        """Get nearest neighbors from an input vector."""
        nns = self._ann_obj.get_nns_by_vector(vec, n, search_k)
        return [self._ann_map_inv[x] for x in nns]
    def get_item_vector(self, entity_id):
        """Get a vector for an entity."""
        if entity_id in self._ann_map:
            return self._ann_obj.get_item_vector(self._ann_map[entity_id])
        else:
            return []
    def __iter__(self):
        """Iterate over object, return (entity_id, vector) tuples."""
        return (EntityVector(
                    entity_id=entity_id,
                    vector=self.get_item_vector(entity_id)
                ) for entity_id in self._ann_map.keys())


class EntitySet(object):
    def __init__(self, nfactor):
        """Initialize EntitySet."""
        self._is_built = False
        self._nfactor = nfactor
        self._annoy_objects = {}
        self._entity_id_map = {}
    def create_entity_type(self, entity_type_id, entity_type,
                           ntrees, metric='angular'):
        """Create an entity type and populate its metadata."""
        self._annoy_objects[entity_type] = EntityType(
            self._nfactor, ntrees, metric, entity_type_id, entity_type)
        self._entity_id_map[entity_type_id] = entity_type
    def add_item(self, entity_type_id, entity_id, factors):
        """Wrap annoy_object add_item."""
        self._annoy_objects[self._entity_id_map[entity_type_id]].add_item(
            entity_id, factors)
    def build(self, parallel=False, verbose=False):
        """Loop through all annoy objects and build in order."""
        # todo: this could be parallelized
        if self._is_built:
            return
        for annoy_object in self._annoy_objects.values():
            logging.info("Starting build for entity {} - {}...".format(
                annoy_object._entity_type_id,
                annoy_object._entity_type))
            annoy_object.build(verbose)
            logging.info("Done build for entity {} - {}".format(
                annoy_object._entity_type_id,
                annoy_object._entity_type))
        self._is_built = True
    def get_vector(self, entity_type, entity_id):
        """Wrap annoy_object get_vector."""
        return self._annoy_objects[entity_type].get_item_vector(entity_id)
    @staticmethod
    def vec_score(vec1, vec2):
        return sum(x*y for x, y in zip(vec1, vec2))
    def get_similar_vector(self, match_vector, match_type, num_similar,
                           oversample):
        """Get similar items from an input vector."""
        if not match_vector:
            return []
        if oversample < 1:
            oversample = 1
        search_k = int(num_similar * self._annoy_objects[match_type]._ntrees *
                       oversample)
        similar_items = self._annoy_objects[match_type].get_nns_by_vector(
            match_vector, num_similar, search_k)
        # compute inner products, and sort
        scores = self.get_scores_vector(
            match_vector, match_type, similar_items)
        scores = sorted(scores, key=lambda k: k['score'], reverse=True)
        return scores[:num_similar]
    def get_similar(self, entity_type, entity_id, match_type, num_similar,
                    oversample=1):
        """Get similar items from an item."""
        match_vector = self.get_vector(entity_type, entity_id)
        return self.get_similar_vector(
            match_vector, match_type, num_similar, oversample)
    def get_scores_vector(self, vector, match_type, match_id_array):
        """Score a vector and an array of matching entities."""
        if not vector:
            return []
        scores = []
        for i in match_id_array:
            match_vector = self.get_vector(match_type, i)
            if match_vector:
                scores.append({
                    'entity_id': i,
                    'score': self.vec_score(vector, match_vector)})
            else:
                scores.append({'entity_id': i, 'score': None})
        return scores
    def get_scores(self, entity_type, entity_id, match_type, match_id_array):
        """Score an item and an array of matching entities."""
        vector = self._annoy_objects[entity_type].get_item_vector(entity_id)
        return self.get_scores_vector(
            vector, match_type, match_id_array)
    def save(self, folder):
        """Save object."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        # annoy objects can't be pickled, so save these separately
        for k, v in self._annoy_objects.items():
            annoy_filepath = os.path.join(folder, '{}.ann'.format(k))
            v._ann_obj.save(annoy_filepath)
        pickle_filepath = os.path.join(folder, 'object.pickle')
        with open(pickle_filepath, 'wb') as handle:
            dill.dump(self, handle)
        enttypes = [{
            'num_entities': etype._ann_obj.get_n_items(),
            'entity_type_id': etype._entity_type_id,
            'entity_type': etype._entity_type,
            'metric':  etype._metric,
            'num_trees': etype._ntrees
        } for k, etype in self._annoy_objects.items()]
        info_file = os.path.join(folder, 'entity_info.json')
        with open(info_file, 'w') as handle:
            json.dump(enttypes, handle)
    @classmethod
    def load(cls, folder):
        """Load object."""
        filepath = os.path.join(folder, 'entity_info.json')
        with open(filepath) as f:
            enttype_info = json.load(f)
        pickle_filepath = os.path.join(folder, 'object.pickle')
        with open(pickle_filepath, 'rb') as f:
            unpickled_class = dill.load(f)
        for k in unpickled_class._annoy_objects:
            annoy_filepath = os.path.join(folder, '{}.ann'.format(k))
            unpickled_class._annoy_objects[k]._ann_obj = AnnoyIndex(
                unpickled_class._nfactor,
                unpickled_class._annoy_objects[k]._metric)
            try:
                unpickled_class._annoy_objects[k]._ann_obj.load(annoy_filepath)
            except IOError as e:
                raise IOError(
                    "Error: cannot load file {0}, which was built "
                    "with the model. '{1}'".format(annoy_filepath, e))
        enttype_sizes = {}
        for enttype in enttype_info:
            enttype_sizes[enttype['entity_type']]= enttype['num_entities']
        for k, annoy_object in unpickled_class._annoy_objects.items():
            if enttype_sizes[annoy_object._entity_type] != annoy_object._nitems:  # NOQA
                raise ValueError(
                    'Entity type {0} should have size {1} '
                    'but actually has size {2}'.format(
                        annoy_object._entity_type,
                        annoy_object._nitems,
                        enttype_sizes[annoy_object._entity_type],
                    )
                )
        for enttype in enttype_info:
            if enttype['entity_type'] not in unpickled_class._annoy_objects:
                raise ValueError(
                    'Entity type {0} exists in model_info.json '
                    'but was not loaded'.format(enttype['entity_type'])
                )
        return unpickled_class 
    @classmethod
    def load_hdfs(cls, folder, SparkFiles):
        """Load object, requires SparkFiles (imported from pyspark), 
        must have sc.addPyFile(path + file) prior to SparkFiles.get(file)."""
        filepath = os.path.join(folder, 'entity_info.json')
        with open(SparkFiles.get("entity_info.json")) as f:
            enttype_info = json.load(f)
        pickle_filepath = os.path.join(folder, 'object.pickle')
        with open(SparkFiles.get("object.pickle")) as f:
            unpickled_class = dill.load(f)
        for k in unpickled_class._annoy_objects:
            annoy_filepath = os.path.join(folder, '{}.ann'.format(k))
            unpickled_class._annoy_objects[k]._ann_obj = AnnoyIndex(
                unpickled_class._nfactor,
                unpickled_class._annoy_objects[k]._metric)
            try:
                unpickled_class._annoy_objects[k]._ann_obj.load(SparkFiles.get("{}.ann".format(k)))
            except IOError as e:
                raise IOError(
                    "Error: cannot load file {0}, which was built "
                    "with the model. '{1}'".format(annoy_filepath, e))
        enttype_sizes = {}
        for enttype in enttype_info:
            enttype_sizes[enttype['entity_type']]= enttype['num_entities']
        for k, annoy_object in unpickled_class._annoy_objects.items():
            if enttype_sizes[annoy_object._entity_type] != annoy_object._nitems:  # NOQA
                raise ValueError(
                    'Entity type {0} should have size {1} '
                    'but actually has size {2}'.format(
                        annoy_object._entity_type,
                        annoy_object._nitems,
                        enttype_sizes[annoy_object._entity_type]))
        for enttype in enttype_info:
            if enttype['entity_type'] not in unpickled_class._annoy_objects:
                raise ValueError(
                    'Entity type {0} exists in model_info.json '
                    'but was not loaded'.format(enttype['entity_type']))
        return unpickled_class
    @classmethod
    def load_hdfs_subset(cls, nfactor, ann_map, SparkFiles):
        """Load subset of ann objects, requires SparkFiles (imported from pyspark), 
        must have sc.addPyFile(path + file) prior to SparkFiles.get(file).
        ann_map type dict format {int:str}"""
        with open(SparkFiles.get("object.pickle")) as f:
            pkl = dill.load(f)
        t = EntitySet(nfactor)
        for ann_idx, ann_item in ann_map.items():
            ent_type = EntityType(nfactor, 50, "angular", ann_idx, ann_item)
            ent_type.__dict__ = pkl._annoy_objects[ann_item].__dict__
            ent_type._ann_obj = AnnoyIndex(100)
            ent_type._ann_obj.load(SparkFiles.get(ann_item + ".ann"))
            t._annoy_objects[ann_item] = ent_type
            t._entity_id_map[ann_idx] = ann_item
        t._is_built=True
        return t
