"""Calculate native lifecycle review on pinned original Motion masks, then attach it."""
import argparse
import json

from pymicroglia.ui.exploration_lifecycle import calculate_lifecycle_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Selected saved measurement run')
    parser.add_argument('--stem', required=True, help='Recording declared in that run')
    parser.add_argument('--tracking-labels', help='Original masks; defaults to the pinned tracking attachment')
    parser.add_argument('--tracking-sha256', help='Expected original mask checksum when supplied explicitly')
    parser.add_argument('--unclaimed', help='Original unclaimed foreground; defaults to the recorded Motion output role')
    args = parser.parse_args()
    print('Checking original tracking masks and measured identity continuity...', flush=True)
    result = calculate_lifecycle_review(args.run, args.stem, tracking_labels=args.tracking_labels,
                                       expected_tracking_sha256=args.tracking_sha256, unclaimed=args.unclaimed)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
