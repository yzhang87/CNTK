//
// <copyright file="BlockRandomizer.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// BlockRandomizer.cpp -- implementation of the block randomizer
//

#include "stdafx.h"
#include "BlockRandomizer.h"

namespace msra { namespace dbn {

    // shuffle a vector into random order by randomly swapping elements
    template<typename VECTOR> static void BlockRandomizer::randomshuffle (VECTOR & v, size_t randomseed)
    {
        if (v.size() > RAND_MAX * (size_t) RAND_MAX)
            RuntimeError("randomshuffle: too large set: need to change to different random generator!");
        srand ((unsigned int) randomseed);
        foreach_index (i, v)
        {
            // pick a random location
            const size_t irand = msra::dbn::rand (0, v.size());

            // swap element i with it
            if (irand == (size_t) i)
                continue;
            ::swap (v[i], v[irand]);
        }
    }

    size_t BlockRandomizer::lazyrandomization(
        const size_t globalts,
        const std::vector<std::vector<utterancechunkdata>> & allchunks)
    {
        // TODO allchunks / utterancechunkdata: wants to know:
        // # chunks, # streams, utterances per chunk, frames per chunk.
        // For checks: length of an utterance
        // For getChunkData - returns reference
        const size_t sweep = globalts / _totalframes;    // which sweep (this determines randomization)
        if (sweep == currentsweep)                       // already got this one--nothing to do
            return sweep;

        currentsweep = sweep;
        if (verbosity > 0)
            fprintf(stderr, "lazyrandomization: re-randomizing for sweep %llu in %s mode\n",
                currentsweep, framemode ? "frame" : "utterance");

        const size_t sweepts = sweep * _totalframes;     // first global frame index for this sweep

        // first randomize chunks
        std::vector<std::vector<std::vector<utterancechunkdata>::const_iterator>> randomizedchunkrefs;
        foreach_index (i, allchunks)
            randomizedchunkrefs.push_back(std::vector<std::vector<utterancechunkdata>::const_iterator>());

        foreach_index (i, allchunks)
            randomizedchunkrefs[i].reserve (allchunks[i].size());

        foreach_index (i, allchunks)    // TODO: this cries for iterating using the iterator!
        {
            foreach_index(j, allchunks[i])
                randomizedchunkrefs[i].push_back (allchunks[i].begin() + j);
            assert (randomizedchunkrefs[i].size() == allchunks[i].size());

            // note that since randomshuffle() uses sweep as seed, this will keep the randomization common across all feature streams
            randomshuffle (randomizedchunkrefs[i], sweep); // bring into random order (with random seed depending on sweep)
        }

        // place them onto the global timeline -> randomizedchunks[]
        // We are processing with randomization within a rolling window over this chunk sequence.
        // Paging will happen on a chunk-by-chunk basis.
        // The global time stamp is needed to determine the paging window.
        randomizedchunks.clear();               // data chunks after being brought into random order (we randomize within a rolling window over them)

        foreach_index(i, allchunks)
            randomizedchunks.push_back(std::vector<chunk>());

        foreach_index(i, allchunks)
        {
            randomizedchunks[i].reserve (randomizedchunkrefs[i].size());
            foreach_index (k, randomizedchunkrefs[i])
                randomizedchunks[i].push_back (chunk (
                    randomizedchunkrefs[i][k],
                    randomizedchunks[i].empty() ? 0 : randomizedchunks[i].back().utteranceposend(),
                    randomizedchunks[i].empty() ? sweepts : randomizedchunks[i].back().globalte()));

            assert (randomizedchunks[i].size() == allchunks[i].size());
            assert (randomizedchunks[i].empty() || (randomizedchunks[i].back().utteranceposend() == numutterances && randomizedchunks[i].back().globalte() == sweepts + _totalframes));
        }

        // for each chunk, compute the randomization range (w.r.t. the randomized chunk sequence)
        foreach_index(i, randomizedchunks)
        {
            // Only required for first feature stream
            if (i != 0)
                continue;

            foreach_index (k, randomizedchunks[i])
            {
                chunk & chunk = randomizedchunks[i][k];
                // start with the range of left neighbor
                if (k == 0)
                {
                    chunk.windowbegin = 0;
                    chunk.windowend = 1;
                }
                else
                {
                    chunk.windowbegin = randomizedchunks[i][k-1].windowbegin;  // might be too early
                    chunk.windowend = randomizedchunks[i][k-1].windowend;      // might have more space
                }
                while (chunk.globalts - randomizedchunks[i][chunk.windowbegin].globalts > randomizationrange/2)
                    chunk.windowbegin++;            // too early
                while (chunk.windowend < randomizedchunks[i].size() && randomizedchunks[i][chunk.windowend].globalte() - chunk.globalts < randomizationrange/2)
                    chunk.windowend++;              // got more space
            }
        }

        // This completes chunk randomization.
        // Now set up the following members for sequence randomization (i.e., utterance or frame):
        //  - positionchunkwindows
        //  - randomizedsequencerefs - this is the data structure being shuffled
        //  - randomizedutteranceposmap

        // TODO adapt comments below. TODO test in utterance mode
        // We will now introduce the concept of utterance *position*.
        // During processing, utterances will be indexed by position (which is in turn derived from a frame index in getbatch()),
        // and it is assumed (required) that positions are requested consecutively.
        // Each utterance position has an underlying associated utterance, which is represented as (chunkid, within-chunk index) and randomly assigned.
        // Each utterance position also has an associated range of chunks that are kept in memory,
        // and the associated underlying utterance is guaranteed to be found within that associated range of chunks.
        // That allows to page out/in data when processing utterance positions in a consecutive manner.

        // compute chunk windows for every utterance position -> positionchunkwindows[]
        // Utterance positions can only reference underlying utterance data within the chunk window.
        // Utterance positions are defined by the randomized chunk sequence (i.e. their underlying 'defining' chunk differs from sweep to sweep).
        size_t numsequences = framemode ? _totalframes : numutterances;

        positionchunkwindows.clear();           // [utterance position] -> [windowbegin, windowend) for controlling paging
        positionchunkwindows.reserve(numsequences);

        // positionchunkwindows should be consistent for all inputs (distinct feature streams), so just build based on feature[0]
        // contains pointer to chunk elements but only to compute index
        foreach_index (k, randomizedchunks[0]) // TODO: this really cries for iterating using iterators!
        {
            chunk & chunk = randomizedchunks[0][k];
            for (size_t i = 0; i < chunk.numutterances(); i++)  // loop over utterances in this chunk
            {
                size_t numsequences = framemode ? chunk.getchunkdata().numframes(i) : 1;
                for (size_t m = 0; m < numsequences; m++)
                {
                    positionchunkwindows.push_back(randomizedchunks[0].begin() + k);
                }
            }
        }
        assert(positionchunkwindows.size() == (framemode ? _totalframes : numutterances));

        // build the randomized utterances array -> randomizedsequencerefs[]
        // start by assigning all utterance positions to utterances in non-random consecutive manner
        randomizedsequencerefs.clear();        // [pos] randomized utterance ids
        randomizedsequencerefs.reserve(numsequences);
        foreach_index (k, randomizedchunks[0])
        {
            chunk & chunk = randomizedchunks[0][k];
            for (size_t i = 0; i < chunk.numutterances(); i++)  // loop over utterances in this chunk
            {
                size_t numsequences = framemode ? chunk.getchunkdata().numframes(i) : 1;
                for (size_t m = 0; m < numsequences; m++)
                {
                    randomizedsequencerefs.push_back (sequenceref /* utteranceref */ (k, i, m));
                }
            }
        }
        assert(randomizedsequencerefs.size() == numsequences);

        // check we got those setup right
        foreach_index (i, randomizedsequencerefs)
        {
            auto & uttref = randomizedsequencerefs[i];
            assert(positionchunkwindows[i].isvalidforthisposition(uttref)); uttref;
        }

        // we now randomly shuffle randomizedsequencerefs[pos], while considering the constraints of what chunk range needs to be in memory
        srand ((unsigned int) sweep + 1);
        for (size_t i = 0; i < randomizedsequencerefs.size(); i++)
        {
            // get valid randomization range, expressed in chunks
            const size_t windowbegin = positionchunkwindows[i].windowbegin();
            const size_t windowend =   positionchunkwindows[i].windowend();

            // get valid randomization range, expressed in utterance positions
            // Remember, utterance positions are defined by chunks.
            size_t posbegin;
            size_t posend;

            // TODO abstract across these (should be sequence indices...)
            if (framemode)
            {
                // in frames
                posbegin = randomizedchunks[0][windowbegin].globalts   - sweepts;
                posend =   randomizedchunks[0][windowend-1].globalte() - sweepts;
            }
            else
            {
                posbegin = randomizedchunks[0][windowbegin].utteranceposbegin;
                posend =   randomizedchunks[0][windowend-1].utteranceposend();
            }

            // randomization range for this utterance position is [posbegin, posend)
            for(;;)
            {
                // pick a random location
                const size_t j = msra::dbn::rand (posbegin, posend);    // a random number within the window
                if (i == j)
                    break;  // the random gods say "this one points to its original position"... nothing wrong about that, but better not try to swap

                // We want to swap utterances at i and j, but need to make sure they remain in their allowed range.
                // This is guaranteed for a so-far untouched utterance, but both i and j may have been touched by a previous swap.

                // We want to use the utterance previously referenced at utterance position j at position i. Is that allowed?
                if (!positionchunkwindows[i].isvalidforthisposition (randomizedsequencerefs[j]))
                    continue;   // nope --try another

                // Likewise may we use the utterance previously referenced at utterance position i at position j?
                if (!positionchunkwindows[j].isvalidforthisposition (randomizedsequencerefs[i]))
                    continue;   // nope --try another

                // yep--swap them
                ::swap (randomizedsequencerefs[i], randomizedsequencerefs[j]); // TODO old swap was perhaps more efficient
                break;
            }
        }

        size_t t = sweepts;
        foreach_index (i, randomizedsequencerefs)
        {
            auto & uttref = randomizedsequencerefs[i];
            uttref.globalts = t;
            if (framemode)
            {
                uttref.numframes = 1;
            }
            else
            {
                uttref.numframes = randomizedchunks[0][uttref.chunkindex].getchunkdata().numframes (uttref.utteranceindex);
            }


            t = uttref.globalte();
        }
        assert (t == sweepts + _totalframes); // TODO does this hold if there we invalid utterance at the end of a chunk?

        // verify that we got it right (I got a knot in my head!)
        foreach_index (i, randomizedsequencerefs)
        {
            // get utterance referenced at this position
            const auto & uttref = randomizedsequencerefs[i];
            // check if it is valid for this position
            if (uttref.chunkindex < positionchunkwindows[i].windowbegin() || uttref.chunkindex >= positionchunkwindows[i].windowend())
                LogicError("lazyrandomization: randomization logic mangled!");
        }

        // create lookup table for (globalts values -> pos) -> randomizedutteranceposmap[]
        randomizedutteranceposmap.clear();      // [globalts] -> pos lookup table
        foreach_index (pos, randomizedsequencerefs)
        {
            auto & uttref = randomizedsequencerefs[pos];
            randomizedutteranceposmap[uttref.globalts] = (size_t) pos;
        }

        // TODO refactor into method
        // check it --my head spins
        t = 0;
        foreach_index (i, randomizedchunks[0])
        {
            const auto & chunk = randomizedchunks[0][i];       // for window and chunkdata
            const size_t poswindowbegin = chunk.windowbegin;
            const size_t poswindowend = chunk.windowend;

            const size_t numutt = chunk.numutterances();
            const auto & chunkdata = chunk.getchunkdata();  // for numframes
            for (size_t k = 0; k < numutt; k++)
            {
                const size_t n = framemode ? chunkdata.numframes(k) : 1;
                for (size_t m = 0; m < n; m++)
                {
                    //const size_t randomizedchunkindex = randomizedframerefs[t].chunkindex;
                    const size_t randomizedchunkindex = randomizedsequencerefs[t].chunkindex;
                    if (randomizedchunkindex < poswindowbegin || randomizedchunkindex >= poswindowend)
                        LogicError("lazyrandomization: nope, you got frame randomization wrong, dude");
                    t++;
                }
            }
        }
        assert (t == numsequences);

        return sweep;
    }

} }
