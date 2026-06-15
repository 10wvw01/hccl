# RFC Document Directory

This directory stores the technical solution design documents (RFC - Request for Comments) of the HCOMM repository.

## RFC Lifecycle

1. **Drafting**: Developers write RFC documents using [the RFC template](./0000-template.md).
2. **Submiting a PR**: Developers submit the RFC as a pull request (PR) to this repository.
3. **Review**: Maintainers review the RFC, which may involve multiple iterations.
4. **Decision-making**
   - **Accepted**: The RFC is approved and can be implemented.
   - **Rejected**: The RFC is rejected, and the PR is closed.
5. **Merge**: Once approved, the RFC is merged and becomes an implementation contract.

## Directory Structure

- `0000-template.md`: RFC template
- `0001-xxxx-xxx.md`: RFC document (ID + short description)

## Naming Rules

RFC file naming format: `{ID}-{short-description}.md`

Example: `0001-add-new-feature.md`

## Related Links

- [Contribution Guide](../CONTRIBUTING.md)
- [SIG Regular Meeting](https://etherpad-cann.meeting.osinfra.cn/p/sig-hccl)
